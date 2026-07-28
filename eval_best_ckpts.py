"""Re-evaluate the best-of-seeds checkpoints from the final all-pairings 1000-iter runs.

Usage:
    python eval_best_ckpts.py                       # scan ~/Downloads/idg_final_bestckpts
    python eval_best_ckpts.py --base <dir>          # scan a different dir for *_bestckpt.{tar,json}
"""

import argparse
import csv
import glob
import json
import os
import tarfile
import tempfile
from datetime import datetime

import ray
from ray.rllib.core.rl_module import RLModule
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune import register_env

from env import GridWorldEnv
from eval_common import (
    always_approve_factory,
    build_inference_module,
    perfect_proposer_factory,
    perfect_validator_factory,
    run_pairing,
    sample_valid_env_variations,
)
from utils import GRID_SIZE, NUM_LAVA_TILES


PAIRINGS = {
    "learned_proposer": dict(
        policy_id="learned_proposer", learned="proposer", deterministic=True,
        display="learned_proposer x perfect_validator",
        counterpart=lambda env: perfect_validator_factory(env)),
    "always_approve": dict(
        policy_id="learned_proposer", learned="proposer", deterministic=True,
        display="learned_proposer x always_approve_validator",
        counterpart=lambda env: always_approve_factory(env)),
    "perfect_proposer": dict(
        policy_id="learned_validator", learned="validator", deterministic=True,
        display="perfect_proposer x learned_validator",
        counterpart=lambda env: perfect_proposer_factory(env)),
    "random_proposer": dict(
        policy_id="learned_validator", learned="validator", deterministic=False,
        display="random_proposer x learned_validator",
        counterpart=lambda env: build_inference_module(env, "proposer", RandomRLModule)),
}

FIELDS = ["algo", "config", "pairing", "pairing_display", "learned_side", "best_seed",
          "recorded_goal_pct", "eval_goal_pct", "goal_wins", "n_configs", "match",
          "validator_mean_reward", "wanted_pct", "good_disobey_pct", "bad_disobey_pct",
          "good_disobey_rel_pct", "bad_disobey_rel_pct", "n_validator_decisions",
          "total_disobey"]


def load_learned(ckpt_tar: str, policy_id: str, workdir: str):
    """Extract the checkpoint tar and load the learned module for `policy_id`."""
    dest = tempfile.mkdtemp(dir=workdir)
    with tarfile.open(ckpt_tar) as t:
        t.extractall(dest)
    seed_dir = os.path.join(dest, os.listdir(dest)[0])  
    rlm = os.path.abspath(os.path.join(seed_dir, "learner_group", "learner",
                                       "rl_module", policy_id))
    if not os.path.isdir(rlm):
        raise FileNotFoundError(f"policy '{policy_id}' not in checkpoint: {rlm}")
    return RLModule.from_checkpoint(rlm)


def eval_one(meta_json: str, variations, workdir: str) -> dict:
    meta = json.loads(open(meta_json).read())
    tar = meta_json[:-5] + ".tar"
    algo = os.path.basename(os.path.dirname(meta_json))
    tag, pairing = meta["tag"], meta["pairing"]
    spec = PAIRINGS[pairing]

    learned = load_learned(tar, spec["policy_id"], workdir)
    if spec["learned"] == "proposer":
        p_factory = lambda env, m=learned: m
        v_factory = spec["counterpart"]
    else:
        p_factory = spec["counterpart"]
        v_factory = lambda env, m=learned: m

    res = run_pairing(f"{algo}_{tag}_{pairing}", p_factory, v_factory, variations,
                      save_video=False, render=False)

    rec = meta.get("goal_pct")
    match = ("n/a (stochastic)" if not spec["deterministic"]
             else ("YES" if abs(res["goal_pct"] - rec) < 0.01 else "MISMATCH"))
    return {
        "algo": algo, "config": tag, "pairing": pairing,
        "pairing_display": spec["display"], "learned_side": spec["learned"],
        "best_seed": meta.get("seed"), "recorded_goal_pct": round(rec, 4),
        "eval_goal_pct": round(res["goal_pct"], 4), "goal_wins": res["goal_wins"],
        "n_configs": res["n_configs"], "match": match,
        "validator_mean_reward": round(res["validator_mean_reward"], 4),
        "wanted_pct": round(res["wanted_pct"], 2),
        "good_disobey_pct": round(res["good_disobey_pct"], 2),
        "bad_disobey_pct": round(res["bad_disobey_pct"], 2),
        "good_disobey_rel_pct": round(res.get("good_disobey_rel_pct", 0.0), 2),
        "bad_disobey_rel_pct": round(res.get("bad_disobey_rel_pct", 0.0), 2),
        "n_validator_decisions": res["n_validator_decisions"],
        "total_disobey": res.get("total_disobey", 0),
    }


PAIRING_ORDER = ["learned_proposer", "always_approve", "perfect_proposer", "random_proposer"]


def make_table(rows: list) -> str:
    cols = [
        ("algo", lambda r: r["algo"]),
        ("config", lambda r: r["config"]),
        ("pairing", lambda r: r["pairing_display"]),
        ("seed", lambda r: str(r["best_seed"])),
        ("goal %", lambda r: f"{r['eval_goal_pct']:6.2f}"),
        ("goal/N", lambda r: f"{r['goal_wins']}/{r['n_configs']}"),
        ("val_reward", lambda r: f"{r['validator_mean_reward']:+.4f}"),
        ("wanted %", lambda r: f"{r['wanted_pct']:6.2f}"),
        ("good/dis %", lambda r: f"{r['good_disobey_rel_pct']:6.2f}"),
        ("bad/dis %", lambda r: f"{r['bad_disobey_rel_pct']:6.2f}"),
        ("decisions", lambda r: str(r["n_validator_decisions"])),
        ("tot_dis", lambda r: str(r["total_disobey"])),
        ("recorded", lambda r: f"{r['recorded_goal_pct']:6.2f}"),
    ]
    order = {p: i for i, p in enumerate(PAIRING_ORDER)}
    rows = sorted(rows, key=lambda r: (order.get(r["pairing"], 9), r["algo"], r["config"]))
    headers = [h for h, _ in cols]
    body = [[fn(r) for _, fn in cols] for r in rows]
    w = [max(len(headers[i]), max((len(b[i]) for b in body), default=0)) for i in range(len(cols))]
    sep = "  ".join("-" * x for x in w)
    out = [sep, "  ".join(headers[i].ljust(w[i]) for i in range(len(cols))), sep]
    out += ["  ".join(b[i].ljust(w[i]) for i in range(len(cols))) for b in body]
    return "\n".join(out + [sep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/Downloads/idg_final_bestckpts"),
                    help="dir scanned recursively for *_bestckpt.json (+ sibling .tar)")
    args = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(args.base, "**", "*_bestckpt.json"), recursive=True))
    if not metas:
        raise SystemExit(f"no *_bestckpt.json under {args.base}")
    print(f"found {len(metas)} best-checkpoint(s) under {args.base}\n")

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env("env", lambda _: GridWorldEnv(GRID_SIZE, num_lava_tiles=NUM_LAVA_TILES,
                                               single_agent=False))
    variations = sample_valid_env_variations(GRID_SIZE, NUM_LAVA_TILES)

    rows, mismatches = [], []
    with tempfile.TemporaryDirectory() as workdir:
        for i, mj in enumerate(metas, 1):
            try:
                r = eval_one(mj, variations, workdir)
            except Exception as e:
                print(f"[{i}/{len(metas)}] {os.path.basename(mj)}  FAILED: {type(e).__name__}: {e}")
                continue
            rows.append(r)
            flag = "" if r["match"] in ("YES", "n/a (stochastic)") else "  <-- MISMATCH!"
            if r["match"] == "MISMATCH":
                mismatches.append(r)
            print(f"[{i}/{len(metas)}] {r['algo']:3} {r['config']:26} {r['pairing']:16} "
                  f"seed{r['best_seed']}  goal {r['eval_goal_pct']:6.2f}% (rec {r['recorded_goal_pct']:6.2f}) "
                  f"{r['match']}{flag}")
    ray.shutdown()

    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"best_ckpt_eval_{ts}.csv")
    txt_path = os.path.join(out_dir, f"best_ckpt_eval_{ts}.txt")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    header = ("=" * 120 + "\n"
              f"BEST-CHECKPOINT EVAL — final all-pairings 1000-iter runs ({len(rows)} checkpoints)\n"
              "Each is the best-of-10-seeds checkpoint per (config, pairing); "
              f"{GRID_SIZE}x{GRID_SIZE} grid, {NUM_LAVA_TILES} lava, N={len(variations)} configs, greedy eval.\n"
              "'recorded' = goal% the seeding notebook logged for this checkpoint. The three\n"
              "deterministic pairings re-evaluate to exactly that value; random_proposer differs\n"
              "run to run because the proposer acts randomly.\n"
              + "=" * 120)
    report = header + "\n" + make_table(rows) + "\n"
    if mismatches:
        report += ("\nWARNING: deterministic-pairing mismatches (loading/pairing bug?):\n"
                   + "\n".join(f"  {m['algo']} {m['config']} {m['pairing']}: "
                              f"eval {m['eval_goal_pct']} vs recorded {m['recorded_goal_pct']}"
                              for m in mismatches) + "\n")
    with open(txt_path, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"CSV: {csv_path}\ntxt: {txt_path}")
    det = [r for r in rows if r["match"] != "n/a (stochastic)"]
    print(f"\nvalidation: {sum(r['match'] == 'YES' for r in det)}/{len(det)} deterministic re-evals matched"
          f"{' — ' + str(len(mismatches)) + ' MISMATCH(es)!' if mismatches else ' (all good)'}")


if __name__ == "__main__":
    main()
