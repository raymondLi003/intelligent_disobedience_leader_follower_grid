"""Multi-seed retraining of the autotune-selected configs (mean +/- std).

find the picked config from the autotune 
and retrain this picked config across different seeds
Evaluate them and find the mean and stdev 

Metric output:
  - learned proposer  (x perfect validator): goal %
  - learned validator (x perfect proposer): validator mean reward (+ good-
    disobedience % and wanted %)

Example Usage:
    python run_seeds.py                             
    python run_seeds.py --algos dqn,sac --seeds 0,1,2
    python run_seeds.py --pairings learned_proposer --iters 1000
"""

import argparse
import csv
import glob
import json
import os
import statistics
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tree
from ray import tune
import ray
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune import register_env

from config import create_rllib_config, get_search_space
from env import GridWorldEnv
from eval_common import (
    always_approve_factory,
    build_inference_module,
    perfect_proposer_factory,
    perfect_validator_factory,
    run_pairing,
    sample_valid_env_variations,
)
from utils import (
    AgentConfig,
    GRID_SIZE,
    LOG_DIR,
    MAX_ENV_STEPS,
    NUM_LAVA_TILES,
    ProposerPolicies,
    TRAINING_ITERATIONS,
    ValidatorPolicies,
)

# the pairings that are included
PAIRINGS = {
    "learned_proposer": (ProposerPolicies.LEARNED, ValidatorPolicies.PERFECT),
    "perfect_proposer": (ProposerPolicies.PERFECT, ValidatorPolicies.LEARNED),
}


def experiment_name(ac: AgentConfig) -> str:
    return (f"{ac.algorithm_name}_{ac.proposer_policy}_{ac.validator_policy}"
            f"__proposer_sees_lava_{ac.proposer_sees_lava}")


def _load_winner_params(exp_name: str, summary_glob: str):
    """find the winner params from the checkpoints."""
    best = (None, None, -1.0)  # (final_params, file, mtime)
    for path in glob.glob(summary_glob):
        try:
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for entry in data:
            if entry.get("experiment") == exp_name and entry.get("final_params"):
                mtime = os.path.getmtime(path)
                if mtime > best[2]:
                    best = (entry["final_params"], path, mtime)
    return best[0], best[1]


def _winner_config(ac: AgentConfig, final_params: dict, seed: int):
    """Rebuild the config based on the chosen config from the checkpoint"""
    cfg = create_rllib_config(ac)
    learns_validator = ac.validator_policy == ValidatorPolicies.LEARNED
    search_keys = set(get_search_space(ac.algorithm_name, learns_validator))
    overrides = {k: final_params[k] for k in search_keys if k in final_params}
    cfg = cfg.update_from_dict(overrides)
    return cfg.debugging(seed=seed)


def _learned_policy_id(ac: AgentConfig) -> str:
    if ac.proposer_policy == ProposerPolicies.LEARNED:
        return ProposerPolicies.LEARNED
    return ValidatorPolicies.LEARNED


def _eval_factories(ac: AgentConfig, algo):
    """Pair the trained learned module against its fixed pairing."""
    learned_module = algo.get_module(_learned_policy_id(ac))

    if ac.proposer_policy == ProposerPolicies.LEARNED:
        p_factory = lambda env, m=learned_module: m
        v_factory = {
            ValidatorPolicies.PERFECT: perfect_validator_factory,
            ValidatorPolicies.ALWAYS_APPROVE: always_approve_factory,
        }[ac.validator_policy]
        return p_factory, v_factory

    # learned validator
    v_factory = lambda env, m=learned_module: m
    if ac.proposer_policy == ProposerPolicies.PERFECT:
        p_factory = perfect_proposer_factory
    elif ac.proposer_policy == ProposerPolicies.RANDOM:
        p_factory = lambda env: build_inference_module(env, "proposer", RandomRLModule)
    else:
        raise ValueError(f"Unsupported proposer for learned validator: {ac.proposer_policy}")
    return p_factory, v_factory


def _primary_metric(ac: AgentConfig) -> str:
    # Goal-reach % for both learned prop and val
    return "goal_pct"


def _retrain_and_eval(ac: AgentConfig, final_params: dict, seed: int, iters: int,
                      variations, num_env_runners, save_ckpt: bool) -> dict:
    """Train the winner config with one seed, then run the deterministic 19-config eval."""
    cfg = _winner_config(ac, final_params, seed)
    if num_env_runners is not None:
        cfg = cfg.env_runners(num_env_runners=num_env_runners)
    algo = cfg.build()
    try:
        for _ in range(iters):
            algo.train()
        p_factory, v_factory = _eval_factories(ac, algo)
        res = run_pairing("seed_eval", p_factory, v_factory, variations,
                          save_video=False, render=False)
        if save_ckpt:
            ckpt_dir = LOG_DIR / "seeds" / experiment_name(ac) / f"seed_{seed}"
            ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
            algo.save_to_path(str(ckpt_dir))
    finally:
        algo.stop()
    return res


def _agg(values):
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", default="dqn,sac,ppo")
    parser.add_argument("--pairings", default=",".join(PAIRINGS),
                        help=f"comma-separated subset of {list(PAIRINGS)}")
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="comma-separated seeds (default 5 seeds)")
    parser.add_argument("--iters", type=int, default=TRAINING_ITERATIONS)
    parser.add_argument("--summary", default=str(LOG_DIR / "tune" / "all_experiments_summary*.json"),
                        help="glob for autotune summaries to read C* from")
    parser.add_argument("--num-env-runners", type=int, default=None,
                        help="override env runners (e.g. 0 for a quick smoke)")
    parser.add_argument("--no-save-ckpt", action="store_true",
                        help="do not save per-seed checkpoints")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    pairing_keys = [p.strip() for p in args.pairings.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    ray.init(ignore_reinit_error=True)
    variations = sample_valid_env_variations(GRID_SIZE, NUM_LAVA_TILES)

    n_runs = len(algos) * len(pairing_keys) * len(seeds)
    print(f"Multi-seed retrain: {len(algos)} algos x {len(pairing_keys)} pairings x "
          f"{len(seeds)} seeds = {n_runs} training runs of {args.iters} iters each.")

    per_seed_rows = []   # one row per (exp, seed)
    summary_rows = []    # one row per exp with mean/std

    for pkey in pairing_keys:
        prop, val = PAIRINGS[pkey]
        for algo in algos:
            ac = AgentConfig(proposer_policy=prop, validator_policy=val, algorithm_name=algo)
            exp = experiment_name(ac)
            final_params, src = _load_winner_params(exp, args.summary)
            if final_params is None:
                print(f"[skip] no winner found for {exp} in {args.summary}")
                continue

            # register the training env (randomized spawn, like run_all_experiments)
            register_env("env", lambda _, ac=ac: GridWorldEnv(
                size=GRID_SIZE, num_lava_tiles=NUM_LAVA_TILES, single_agent=False,
                max_steps=MAX_ENV_STEPS, proposer_sees_lava=ac.proposer_sees_lava,
                randomize_spawn=True,
            ))

            metric = _primary_metric(ac)
            print(f"\n=== {exp}  (C* from {Path(src).name}, metric={metric}) ===")
            vals = []
            for seed in seeds:
                res = _retrain_and_eval(ac, final_params, seed, args.iters, variations,
                                        args.num_env_runners, save_ckpt=not args.no_save_ckpt)
                row = {
                    "experiment": exp, "algo": algo, "pairing": pkey, "seed": seed,
                    "goal_pct": res["goal_pct"],
                    "validator_mean_reward": res["validator_mean_reward"],
                    "good_disobey_rel_pct": res.get("good_disobey_rel_pct", 0.0),
                    "wanted_pct": res["wanted_pct"],
                }
                per_seed_rows.append(row)
                vals.append(res[metric])
                print(f"  seed {seed}: {metric}={res[metric]:.4f}  "
                      f"(goal%={res['goal_pct']:.2f} val_reward={res['validator_mean_reward']:+.4f})")

            mean, std = _agg(vals)
            print(f"  --> {metric} = {mean:.4f} +/- {std:.4f}  (n={len(vals)} seeds)")
            summary_rows.append({
                "experiment": exp, "algo": algo, "pairing": pkey, "metric": metric,
                "mean": mean, "std": std, "n_seeds": len(vals),
                "seed_values": ";".join(f"{v:.4f}" for v in vals),
            })

    out_dir = LOG_DIR.parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if per_seed_rows:
        per_path = out_dir / f"seed_runs_perseed_{ts}.csv"
        with per_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            w.writeheader(); w.writerows(per_seed_rows)
        sum_path = out_dir / f"seed_runs_summary_{ts}.csv"
        with sum_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)

        print("\n" + "=" * 72)
        print("MEAN +/- STD ACROSS SEEDS")
        print("=" * 72)
        for r in summary_rows:
            print(f"  {r['experiment']:<60}  {r['metric']:<22} {r['mean']:.4f} +/- {r['std']:.4f}  (n={r['n_seeds']})")
        print(f"\nPer-seed CSV: {per_path}\nSummary CSV:  {sum_path}")
    else:
        print("No runs completed, there's no winner config found.")

    ray.shutdown()


if __name__ == "__main__":
    main()
