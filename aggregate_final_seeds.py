"""Aggregate per-seed CSVs from the final all-pairings 1000-iter runs into mean +/- std per (algo, config, pairing).

Usage:
    python aggregate_final_seeds.py                  # scan the unzipped final folders
    python aggregate_final_seeds.py --base <dir>
"""

import argparse
import csv
import glob
import os
import re
import statistics
from datetime import datetime

PAIRINGS = {
    "learned_proposer": "learned_proposer x perfect_validator",
    "always_approve": "learned_proposer x always_approve_validator",
    "perfect_proposer": "perfect_proposer x learned_validator",
    "random_proposer": "random_proposer x learned_validator",
}
PAIRING_ORDER = list(PAIRINGS)
N_CONFIGS = 19

FIELDS = ["algo", "config", "pairing", "pairing_display", "n_seeds",
          "goal_pct_mean", "goal_pct_std", "goal_pct_min", "goal_pct_max",
          "goal_wins_mean", "n_configs",
          "validator_mean_reward_mean", "validator_mean_reward_std",
          "wanted_pct_mean", "wanted_pct_std",
          "good_disobey_rel_pct_mean", "good_disobey_rel_pct_std",
          "bad_disobey_rel_pct_mean", "bad_disobey_rel_pct_std",
          "per_seed_goal_pct"]


def parse_tag(fname: str, algo: str, pairing: str) -> str:
    """Return the tag, the piece between 'final_<algo>_' and '_<pairing>_i1000_seed<k>'."""
    m = re.match(rf"final_{re.escape(algo)}_(.+)_{re.escape(pairing)}_i1000_seed\d+\.csv$", fname)
    if not m:
        raise ValueError(f"cannot parse tag from {fname}")
    return m.group(1)


def collect(base: str) -> dict:
    """(algo, tag, pairing) -> {seed: row}"""
    groups = {}
    for path in glob.glob(os.path.join(base, "**", "*_i1000_seed*.csv"), recursive=True):
        for r in csv.DictReader(open(path)):
            algo, pairing, seed = r["algo"], r["pairing"], int(r["seed"])
            if pairing not in PAIRINGS:
                continue
            tag = parse_tag(os.path.basename(path), algo, pairing)
            groups.setdefault((algo, tag, pairing), {})[seed] = r
    return groups


def ms(vals):
    """mean, sample std (0.0 for a single observation)."""
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def summarize(groups: dict) -> list:
    rows = []
    for (algo, tag, pairing), by_seed in groups.items():
        seeds = sorted(by_seed)
        goal = [float(by_seed[s]["goal_pct"]) for s in seeds]
        rew = [float(by_seed[s]["validator_mean_reward"]) for s in seeds]
        want = [float(by_seed[s]["wanted_pct"]) for s in seeds]
        good = [float(by_seed[s]["good_disobey_rel_pct"]) for s in seeds]
        # exact identity; always_approve never disobeys -> both shares are 0
        bad = [0.0 if pairing == "always_approve" else 100.0 - g for g in good]
        g_m, g_s = ms(goal); r_m, r_s = ms(rew); w_m, w_s = ms(want)
        gd_m, gd_s = ms(good); bd_m, bd_s = ms(bad)
        rows.append({
            "algo": algo, "config": tag, "pairing": pairing,
            "pairing_display": PAIRINGS[pairing], "n_seeds": len(seeds),
            "goal_pct_mean": round(g_m, 2), "goal_pct_std": round(g_s, 2),
            "goal_pct_min": round(min(goal), 2), "goal_pct_max": round(max(goal), 2),
            "goal_wins_mean": round(g_m * N_CONFIGS / 100, 1), "n_configs": N_CONFIGS,
            "validator_mean_reward_mean": round(r_m, 4), "validator_mean_reward_std": round(r_s, 4),
            "wanted_pct_mean": round(w_m, 2), "wanted_pct_std": round(w_s, 2),
            "good_disobey_rel_pct_mean": round(gd_m, 2), "good_disobey_rel_pct_std": round(gd_s, 2),
            "bad_disobey_rel_pct_mean": round(bd_m, 2), "bad_disobey_rel_pct_std": round(bd_s, 2),
            "per_seed_goal_pct": ",".join(f"{g:.0f}" for g in goal),
        })
    return rows


def make_table(rows: list) -> str:
    cols = [
        ("algo", lambda r: r["algo"]),
        ("config", lambda r: r["config"]),
        ("pairing", lambda r: r["pairing_display"]),
        ("n", lambda r: str(r["n_seeds"])),
        ("goal %", lambda r: f"{r['goal_pct_mean']:6.2f}+/-{r['goal_pct_std']:5.2f}"),
        ("goal/N", lambda r: f"{r['goal_wins_mean']:4.1f}/{r['n_configs']}"),
        ("val_reward",
         lambda r: f"{r['validator_mean_reward_mean']:+.4f}+/-{r['validator_mean_reward_std']:.4f}"),
        ("wanted %", lambda r: f"{r['wanted_pct_mean']:6.2f}+/-{r['wanted_pct_std']:5.2f}"),
        ("good/dis %",
         lambda r: f"{r['good_disobey_rel_pct_mean']:6.2f}+/-{r['good_disobey_rel_pct_std']:5.2f}"),
        ("bad/dis %",
         lambda r: f"{r['bad_disobey_rel_pct_mean']:6.2f}+/-{r['bad_disobey_rel_pct_std']:5.2f}"),
        ("best", lambda r: f"{r['goal_pct_max']:6.2f}"),
        ("per-seed goal % (seed order)", lambda r: r["per_seed_goal_pct"]),
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
    ap.add_argument("--base", default=os.path.expanduser("~/Downloads/idg_final_seeds"),
                    help="dir scanned recursively for final_*_i1000_seed*.csv "
                         "(the unzipped idg_final_<algo>_1000iters folders)")
    args = ap.parse_args()

    groups = collect(args.base)
    if not groups:
        raise SystemExit(f"no per-seed CSVs under {args.base}")
    rows = summarize(groups)
    n_files = sum(len(v) for v in groups.values())

    counts = sorted({r["n_seeds"] for r in rows})
    header = ("=" * 150 + "\n"
              f"FINAL ALL-PAIRINGS SEED SUMMARY — {len(rows)} config-pairings x "
              f"{'/'.join(map(str, counts))} seeds ({n_files} runs), 1000 iters\n"
              "values are across-seed mean +/- std (ddof=1); 'best' is the top seed's goal %\n"
              + "=" * 150)
    report = header + "\n" + make_table(rows) + "\n"

    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"final_seed_summary_{ts}.csv")
    txt_path = os.path.join(out_dir, f"final_seed_summary_{ts}.txt")
    order = {p: i for i, p in enumerate(PAIRING_ORDER)}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (order.get(r["pairing"], 9), r["algo"], r["config"])))
    with open(txt_path, "w") as f:
        f.write(report)

    print(report)
    print(f"CSV: {csv_path}\ntxt: {txt_path}")


if __name__ == "__main__":
    main()
