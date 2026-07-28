"""Multi-seed eval of the LLM validators (perfect proposer x LLM), one temperature per seed.

Usage:
    python run_llm_seeds.py                        # all 4 families (11 models) x seeds 0-9
    python run_llm_seeds.py --llm-families claude  # one provider family only
    python run_llm_seeds.py --models gpt-5-mini    # specific model(s)
    python run_llm_seeds.py --heal-only            # just purge poisoned rows and exit
"""

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import time
from datetime import datetime

import llm_validator_no_strat as lvm
from llm_validator_no_strat import LLMValidatorNoStratExplain, _slugify
from llmproxy import LLMProxy

OUT_DIR = os.path.join("eval_results", "llm_explain_seeds")
PERSEED_CSV = os.path.join(OUT_DIR, "perseed.csv")


def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# seed k -> TEMPERATURES[k]; 0.0-0.9 is valid for every provider
TEMPERATURES = [round(0.1 * k, 1) for k in range(10)]


def seed_temperature(seed: int) -> float:
    """Temperature for a seed index. Seeds past the schedule reuse the last entry."""
    return TEMPERATURES[seed] if seed < len(TEMPERATURES) else TEMPERATURES[-1]


FIELDS = ["model", "seed", "temperature", "goal_pct", "goal_wins", "n_configs",
          "validator_mean_reward", "wanted_pct", "good_disobey", "bad_disobey",
          "good_disobey_rel_pct", "bad_disobey_rel_pct", "total_disobey",
          "n_validator_decisions", "llm_calls", "llm_cache_hits", "elapsed_s", "status"]



QUOTA_MARKERS = ("quota", "rate limit", "rate-limit", "too many requests", "429",
                 "budget", "insufficient", "limit exceeded", "quota exceeded")


class ProxyCallError(RuntimeError):
    """A failed proxy call, with .quota marking quota/rate-limit exhaustion."""

    def __init__(self, message: str, quota: bool):
        super().__init__(message)
        self.quota = quota


class StrictProxy:
    """Wraps LLMProxy so proxy errors raise instead of returning an error dict."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, **kwargs):
        r = self._inner.generate(**kwargs)
        if not isinstance(r, dict) or "error" in r:
            err = str(r.get("error", r)) if isinstance(r, dict) else str(r)
            quota = ((isinstance(r, dict) and r.get("status_code") == 429)
                     or any(m in err.lower() for m in QUOTA_MARKERS))
            raise ProxyCallError(err, quota)
        return r


def make_explain_class(model_name: str, seed: int) -> type:
    """Build an explain-mode validator class pinned to a model and this seed's temperature."""

    def _ensure_proxy(self):
        if self._proxy is None:
            self._proxy = StrictProxy(LLMProxy())
        return self._proxy

    return type(
        f"LLMValidatorExplain_s{seed}_{_slugify(model_name)}",
        (LLMValidatorNoStratExplain,),
        {"MODEL_NAME": model_name, "LOG_TAG": f"explain_s{seed}",
         "TEMPERATURE": seed_temperature(seed), "_ensure_proxy": _ensure_proxy},
    )



# per-seed log paths
def seed_log_paths(model: str, seed: int) -> list:
    slug = _slugify(model)
    tag = f"explain_s{seed}"
    return [str(lvm._LOG_DIR / f"llm_eval_{tag}__{slug}.jsonl"),
            str(lvm._LOG_DIR / f"llm_explain_{tag}__{slug}.jsonl")]


def fresh_logs(model: str, seed: int) -> None:
    for p in seed_log_paths(model, seed):
        if os.path.exists(p):
            os.remove(p)


def poison_stats(model: str, seed: int):
    """Return (n_calls, n_empty, trailing_empty) from the reasoning log."""
    path = seed_log_paths(model, seed)[1]
    if not os.path.exists(path):
        return 0, 0, 0
    flags = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            flags.append((not rec.get("strategy")) and (not rec.get("thought_process"))
                         and rec.get("decision") == 0)
    trail = 0
    for v in reversed(flags):
        if not v:
            break
        trail += 1
    return len(flags), sum(flags), trail


def looks_poisoned(n: int, empty: int, trail: int) -> bool:
    # heuristic: if >20% of calls are empty and the last 3+ are empty, it's poisoned
    return n > 0 and (trail >= 3 or empty / n > 0.2)


def heal(csv_path: str) -> set:
    """Drop 'ok' rows whose logs carry the quota-poison signature and return the healed set."""
    if not os.path.exists(csv_path):
        return set()
    rows = list(csv.DictReader(open(csv_path)))
    keep, healed = [], set()
    for r in rows:
        if r.get("status") == "ok":
            n, empty, trail = poison_stats(r["model"], int(r["seed"]))
            if looks_poisoned(n, empty, trail):
                healed.add((r["model"], int(r["seed"])))
                fresh_logs(r["model"], int(r["seed"]))
                continue
        keep.append(r)
    if healed:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in keep:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        for m, s in sorted(healed):
            print(f"  healed (quota-poisoned, will rerun): {m} seed {s}")
    return healed



# load done and stale sets
def load_done(path: str):
    """Return (done, stale) sets, splitting ok rows by whether their temperature still matches their seed."""
    done, stale = set(), set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                seed = int(r["seed"])
                recorded = float(r["temperature"]) if r.get("temperature") else 0.0
                target = done if abs(recorded - seed_temperature(seed)) < 1e-9 else stale
                target.add((r["model"], seed))
    return done, stale


def drop_rows(csv_path: str, keys: set) -> None:
    """Remove every row for these (model, seed) pairs and delete their logs so they rerun."""
    if not keys or not os.path.exists(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path)))
    keep = [r for r in rows if (r["model"], int(r["seed"])) not in keys]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    for m, s in sorted(keys):
        fresh_logs(m, s)


def append_row(path: str, row: dict) -> None:
    # append a row to the CSV, creating it if needed
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def _model_rank() -> dict:
    """Map each model id to its position in the canonical lineup (empty dict on failure)."""
    try:
        from run_all_eval import LLM_FAMILIES, select_llm_models
        order = [mid for _, mid in select_llm_models(list(LLM_FAMILIES))]
        return {m: i for i, m in enumerate(order)}
    except Exception:
        return {}


def regroup_csv(path: str) -> None:
    """Rewrite the per-seed CSV grouped by model in lineup order, then by seed."""
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    rank = _model_rank()
    tail = len(rank)
    rows.sort(key=lambda r: (rank.get(r["model"], tail), r["model"], int(r["seed"])))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def fmt_table(cols, recs) -> str:
    """Render a fixed-width table matching the eval results layout."""
    headers = [h for h, _ in cols]
    body = [[fn(r) for _, fn in cols] for r in recs]
    w = [max(len(headers[i]), max((len(b[i]) for b in body), default=0)) for i in range(len(cols))]
    sep = "  ".join("-" * x for x in w)
    out = [sep, "  ".join(headers[i].ljust(w[i]) for i in range(len(cols))), sep]
    out += ["  ".join(b[i].ljust(w[i]) for i in range(len(cols))) for b in body]
    return "\n".join(out + [sep])


def summarize(path: str) -> str:
    by_model = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            by_model.setdefault(r["model"], []).append(r)
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
    recs = []
    for model, rows in sorted(by_model.items()):
        goals = [float(r["goal_pct"]) for r in rows]
        wanted = [float(r["wanted_pct"]) for r in rows]
        rew = [float(r["validator_mean_reward"]) for r in rows]
        goodrel = [float(r["good_disobey_rel_pct"]) for r in rows]
        badrel = [float(r["bad_disobey_rel_pct"]) for r in rows]
        by_seed = {int(r["seed"]): float(r["goal_pct"]) for r in rows}
        recs.append(dict(
            model=model, n=len(rows),
            gmean=statistics.mean(goals), gstd=sd(goals),
            gmin=min(goals), gmax=max(goals),
            wanted=statistics.mean(wanted), wanted_std=sd(wanted),
            goodrel=statistics.mean(goodrel), goodrel_std=sd(goodrel),
            badrel=statistics.mean(badrel), badrel_std=sd(badrel),
            rew=statistics.mean(rew),
            # one slot per scheduled temperature; '-' where that seed has not run
            per_t=",".join(f"{by_seed[k]:.0f}" if k in by_seed else "-"
                           for k in range(len(TEMPERATURES))),
        ))
    cols = [
        ("model",      lambda r: r["model"]),
        ("n",          lambda r: str(r["n"])),
        ("goal %",     lambda r: f"{r['gmean']:6.2f}+/-{r['gstd']:5.2f}"),
        ("min",        lambda r: f"{r['gmin']:6.2f}"),
        ("max",        lambda r: f"{r['gmax']:6.2f}"),
        ("wanted %",   lambda r: f"{r['wanted']:6.2f}+/-{r['wanted_std']:5.2f}"),
        ("good/dis %", lambda r: f"{r['goodrel']:6.2f}+/-{r['goodrel_std']:5.2f}"),
        ("bad/dis %",  lambda r: f"{r['badrel']:6.2f}+/-{r['badrel_std']:5.2f}"),
        ("r_F mean",   lambda r: f"{r['rew']:+.4f}"),
        (f"goal % @ t={TEMPERATURES[0]:.1f}..{TEMPERATURES[-1]:.1f}", lambda r: r["per_t"]),
    ]
    header = ("=" * 130 + "\n"
              f"LLM validators (perfect proposer x LLM, EXPLAIN mode) — {len(recs)} model(s), "
              f"{len(TEMPERATURES)}-point temperature sweep\n"
              f"Seed k runs at temperature {TEMPERATURES[0]:.1f}..{TEMPERATURES[-1]:.1f} "
              "(step 0.1) on ONE shared eval suite, so mean +/- std is the spread ACROSS\n"
              "TEMPERATURES -- how sensitive the follower's behaviour is to sampling -- not "
              "run-to-run noise at a fixed temperature.\n"
              "Percentages are sample std (ddof=1). The trailing column lists goal % at each "
              "temperature in order; '-' = not yet run.\n" + "=" * 130)
    return header + "\n" + fmt_table(cols, recs) + "\n"


# main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-families", default="llama,gpt,gemini,claude",
                   help="comma-separated provider families (llama,gpt,gemini,claude)")
    p.add_argument("--models", default=None,
                   help="comma-separated substrings; only run models whose id matches one")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                   help=f"comma-separated seeds; seed k runs at temperature {TEMPERATURES}")
    p.add_argument("--max-configs", type=int, default=None,
                   help="subsample this many lava configs (same suite for every run)")
    p.add_argument("--no-resume", action="store_true", help="rerun even if a row already exists")
    p.add_argument("--no-heal", action="store_true", help="skip the poisoned-row startup check")
    p.add_argument("--heal-only", action="store_true",
                   help="purge quota-poisoned rows from the CSV and exit (no API calls)")
    p.add_argument("--max-consecutive-failures", type=int, default=2,
                   help="stop the sweep after this many consecutive failed runs")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.no_heal:
        healed = heal(PERSEED_CSV)
        if healed:
            print(f"{len(healed)} poisoned run(s) invalidated; they will rerun\n")

    # load the done/stale sets and purge stale rows so they rerun
    done, stale = load_done(PERSEED_CSV)
    if stale:
        # archive before discarding so superseded measurements stay on disk
        backup = os.path.join(OUT_DIR, f"perseed_superseded_{ts_now()}.csv")
        shutil.copy2(PERSEED_CSV, backup)
        drop_rows(PERSEED_CSV, stale)
        print(f"{len(stale)} row(s) were recorded at a temperature their seed no longer maps "
              f"to; they will rerun.\n  previous file archived: {backup}\n")
    if args.no_resume:
        done = set()

    if args.heal_only:
        regroup_csv(PERSEED_CSV)          # regroup the CSV so it's easier to read
        print("heal done (heal-only mode; no API calls made; CSV regrouped by model)")
        return

    # heavy imports only when actually running evals
    import ray
    from ray.tune import register_env
    from env import GridWorldEnv
    from eval_common import (build_inference_module, perfect_proposer_factory,
                             run_pairing, sample_valid_env_variations)
    from run_all_eval import select_llm_models
    from utils import GRID_SIZE, NUM_LAVA_TILES

    families = [x.strip() for x in args.llm_families.split(",") if x.strip()]
    models = select_llm_models(families)
    if args.models:
        needles = [s.strip() for s in args.models.split(",") if s.strip()]
        models = [m for m in models if any(n in m[1] or n in m[0] for n in needles)]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    if not models:
        raise SystemExit("no models selected")

    ray.init(ignore_reinit_error=True)
    register_env("env", lambda _: GridWorldEnv(GRID_SIZE, num_lava_tiles=NUM_LAVA_TILES,
                                               single_agent=False))

    # sample the same set of valid lava configs for every run, so the temperature sweep is apples-to-apples
    variations = sample_valid_env_variations(GRID_SIZE, NUM_LAVA_TILES)
    if args.max_configs and args.max_configs < len(variations):
        variations = random.Random(1).sample(variations, args.max_configs)

    todo = [(disp, mid, s) for (disp, mid) in models for s in seeds if (mid, s) not in done]
    print(f"{len(models)} model(s) x {len(seeds)} seed(s) | {len(variations)} configs each")
    print(f"temperatures: {[seed_temperature(s) for s in seeds]}")
    print(f"{len(todo)} run(s) to do ({len(models) * len(seeds) - len(todo)} already done)\n")

    stop_reason = None
    consecutive_fails = 0
    n_ok = 0
    for i, (disp, model_id, seed) in enumerate(todo, 1):
        temp = seed_temperature(seed)
        name = f"perfect_x_llm_{disp}__explain_s{seed}"
        print(f"[{i}/{len(todo)}] {name}  (t={temp})")
        fresh_logs(model_id, seed)  # aborted earlier attempts must not linger
        t0 = time.time()
        try:
            vc = make_explain_class(model_id, seed)
            res = run_pairing(
                name=name,
                proposer_factory=perfect_proposer_factory,
                validator_factory=lambda env, _vc=vc: build_inference_module(env, "validator", _vc),
                variations=variations,
                save_video=False,
                render=False,
            )
        except ProxyCallError as e:
            fresh_logs(model_id, seed)  # scrub the partial attempt
            if e.quota:
                stop_reason = f"QUOTA EXCEEDED: {e}"
                break
            consecutive_fails += 1
            append_row(PERSEED_CSV, dict(model=model_id, seed=seed, temperature=temp,
                                         status=f"FAILED: {e}"[:200],
                                         elapsed_s=round(time.time() - t0, 1)))
            print(f"    proxy FAILED ({consecutive_fails} in a row): {e}")
            if consecutive_fails >= args.max_consecutive_failures:
                stop_reason = f"{consecutive_fails} consecutive proxy failures (last: {e})"
                break
            continue
        except Exception as e:  # non-proxy problem: record and keep going
            append_row(PERSEED_CSV, dict(model=model_id, seed=seed, temperature=temp,
                                         status=f"FAILED: {type(e).__name__}: {e}"[:200],
                                         elapsed_s=round(time.time() - t0, 1)))
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue

        # never record a run whose log shows poisoning
        n, empty, trail = poison_stats(model_id, seed)
        if looks_poisoned(n, empty, trail):
            fresh_logs(model_id, seed)
            stop_reason = (f"run completed but its log shows {empty}/{n} empty-reasoning "
                           f"calls (trailing {trail}) -- treating as quota poisoning")
            break

        consecutive_fails = 0
        n_ok += 1
        row = {k: res.get(k, "") for k in FIELDS}
        row.update(model=model_id, seed=seed, temperature=temp, status="ok",
                   elapsed_s=round(time.time() - t0, 1))
        append_row(PERSEED_CSV, row)
        print(f"    t={temp}  goal {res['goal_pct']:.2f}%  wanted {res['wanted_pct']:.2f}%  "
              f"calls {res.get('llm_calls', '?')}  ({row['elapsed_s']}s)")

    ray.shutdown()

    if stop_reason:
        remaining = len(todo) - n_ok
        print("\n" + "!" * 78)
        print(f"SWEEP STOPPED: {stop_reason}")
        print(f"{n_ok} run(s) completed this session; ~{remaining} of this selection remain.")
        print("Nothing partial was recorded. When the quota refreshes, rerun the SAME")
        print("command -- finished (model, seed) pairs are skipped automatically.")
        print("!" * 78)

    if os.path.exists(PERSEED_CSV):
        regroup_csv(PERSEED_CSV)          # leave the file grouped by model, not run order
        report = summarize(PERSEED_CSV)
        print("\n" + report)
        ts = ts_now()
        with open(os.path.join(OUT_DIR, f"summary_{ts}.txt"), "w") as f:
            f.write(report)
        print(f"per-seed CSV: {PERSEED_CSV}")
        print(f"summary:      {os.path.join(OUT_DIR, f'summary_{ts}.txt')}")


if __name__ == "__main__":
    main()
