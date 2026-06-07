"""Train every combination for the IDG experiments.

Trains 4 pairings x 3 algorithms = 12 models on a 3x3 grid with 2 lava tiles:

  - Random proposer    x Learned validator    (validator learns)
  - Learned proposer   x Always-approve validator  (proposer learns)
  - Learned proposer   x Perfect validator    (proposer learns)
  - Perfect proposer   x Learned validator    (validator learns)

Each pairing is trained under DQN, PPO, and SAC.

We do not train the LLM, so only in the eval we include the llm validator

After the training completion, a summary mapping each experiment to its search space, 
final hyperparameter configs, and final metrics is saved to: 
logs/tune/all_experiments_summary.json.

Usage:
    python run_all_experiments.py                # full run 
    python run_all_experiments.py --iters 200    # if want to customize iterations
    python run_all_experiments.py --samples 16    # activate autotune and run 16 samples
"""

import argparse
import json
import os
from pathlib import Path

# Disable Ray's new AIR progress output BEFORE importing ray
# otherwise the AIR reporter dumps the entire dict in the output 
os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")

import ray
from ray import tune
from ray.tune import Checkpoint, register_env

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from config import create_rllib_config, get_search_space
from env import GridWorldEnv
from metrics import ActionLoggerCallback, CustomTBXLoggerCallback
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


# proposer_policy & validator_policy
# LEARNED is the side that gets trained.
PAIRINGS = [
    (ProposerPolicies.RANDOM, ValidatorPolicies.LEARNED),
    (ProposerPolicies.LEARNED, ValidatorPolicies.ALWAYS_APPROVE),
    (ProposerPolicies.LEARNED, ValidatorPolicies.PERFECT),
    (ProposerPolicies.PERFECT, ValidatorPolicies.LEARNED),
]

ALGORITHMS = ["dqn", "ppo", "sac"]


def experiment_name(agent_config: AgentConfig) -> str:
    return (
        f"{agent_config.algorithm_name}"
        f"_{agent_config.proposer_policy}_{agent_config.validator_policy}"
        f"__proposer_sees_lava_{agent_config.proposer_sees_lava}"
    )


def _jsonable(obj):
    """convert an RLlib config dict to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


def metric_for(agent_config: AgentConfig) -> str:
    """Per-policy reward metric to optimize during autotune.

    Multi-agent episode_return_mean sums proposer's and validator's rewards, which can be inflated
    by the validator harvesting reward from a bad proposer.
    Optimize the reward of whichever side is actually learning.

    There are two episode return means
      env_runners/agent_episode_returns_mean/<agent_id>     (keyed by agent: proposer/validator)
      env_runners/module_episode_returns_mean/<policy_name> (keyed by policy: learned_proposer/learned_validator)
    We want the policy-keyed one so we always pick the LEARNED side.
    """
    if agent_config.proposer_policy == ProposerPolicies.LEARNED:
        return "env_runners/module_episode_returns_mean/learned_proposer"
    if agent_config.validator_policy == ValidatorPolicies.LEARNED:
        return "env_runners/module_episode_returns_mean/learned_validator"
    return "env_runners/episode_return_mean"


def scheduler_for(algorithm_name: str, iters: int) -> ASHAScheduler:
    """
        change the early stopping to 50 iterations as start
    """
    return ASHAScheduler(
        max_t=iters,
        grace_period=min(50, iters),
        reduction_factor=4,
    )


def train_one(agent_config: AgentConfig, iters: int, samples: int) -> dict:
    config = create_rllib_config(agent_config)
    config.callbacks([ActionLoggerCallback])

    name = experiment_name(agent_config)
    param_space = config.to_dict()

    tune_config = None
    progress_reporter = None
    if samples > 1:
        # activate autotune and include which policy is getting trained
        learns_validator = agent_config.validator_policy == ValidatorPolicies.LEARNED
        search_space = get_search_space(agent_config.algorithm_name, learns_validator=learns_validator)
        for key, sampler in search_space.items():
            param_space[key] = sampler
        tune_config = tune.TuneConfig(
            num_samples=samples,
            metric=metric_for(agent_config),
            mode="max",
            scheduler=scheduler_for(agent_config.algorithm_name, iters),
        )
        # Custom reporter to report only the necessary params
        progress_reporter = CLIReporter(
            parameter_columns=list(search_space.keys()),
            metric_columns={
                metric_for(agent_config): "reward",
                "training_iteration": "iter",
                "time_total_s": "time(s)",
            },
            max_report_frequency=30,
        )

    tuner = tune.Tuner(
        config.algo_class,
        param_space=param_space,
        tune_config=tune_config,
        run_config=tune.RunConfig(
            stop={"training_iteration": iters},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=50,
            ),
            storage_path=LOG_DIR / "tune",
            name=name,
            callbacks=[CustomTBXLoggerCallback()],
            progress_reporter=progress_reporter,
            # 1 = only status table. the default 3 prints the full per-iter result
            verbose=1,
        ),
    )

    results = tuner.fit()
    # In autotune mode, pick the best trial by the per-policy metric we set
    # and it only picks the ones with a full run
    # In single-trial mode, this just returns the only trial.
    if samples > 1:
        metric = metric_for(agent_config)

        def _metric_val(r):
            md = r.metrics or {}
            if metric in md:                      
                v = md[metric]
            else:                                 
                v = md
                for part in metric.split("/"):
                    v = v.get(part) if isinstance(v, dict) else None
            return v if isinstance(v, (int, float)) else float("-inf")

        def _iters(r):
            return (r.metrics or {}).get("training_iteration", 0)

        full_length = [r for r in results if _iters(r) >= iters]
        if not full_length:
            print(f"  warning: no trial reached {iters} iters; falling back to all trials")
            full_length = list(results)
        best = max(full_length, key=_metric_val)
    else:
        best = results.get_best_result()

    if best.checkpoint is not None:
        best.checkpoint.to_directory(LOG_DIR / "tune" / name / "best_checkpoint")

    return {
        "experiment": name,
        "agent_config": {
            "proposer_policy": str(agent_config.proposer_policy),
            "validator_policy": str(agent_config.validator_policy),
            "algorithm_name": agent_config.algorithm_name,
            "proposer_sees_lava": agent_config.proposer_sees_lava,
        },
        "num_samples": samples,
        "metric_optimized": metric_for(agent_config) if samples > 1 else None,
        "final_params": _jsonable(best.config),
        "search_space": _jsonable(param_space),
        "final_metrics": _jsonable(best.metrics) if best.metrics else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=TRAINING_ITERATIONS,
        help=f"training iterations per experiment (default: {TRAINING_ITERATIONS})",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="takes in a comma-separated substring. "
        " the command will only run the experiments that is named in the substring."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Ray Tune samples per experiment. 1 = single fixed config."
             ">1 turns on autotune over the per-algorithm search."
             "the search space is defined in config.get_search_space, with ASHA scheduler for early stopping"

    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    agent_configs = [
        AgentConfig(
            proposer_policy=prop,
            validator_policy=val,
            algorithm_name=algo,
        )
        for algo in ALGORITHMS
        for (prop, val) in PAIRINGS
    ]

    if args.only:
        needles = [s.strip() for s in args.only.split(",") if s.strip()]
        agent_configs = [
            ac for ac in agent_configs
            if any(n in experiment_name(ac) for n in needles)
        ]
        print(f"--only matched {len(agent_configs)} experiment(s):")
        for ac in agent_configs:
            print(f"  - {experiment_name(ac)}")
        if not agent_configs:
            print("Nothing to run. Exiting.")
            ray.shutdown()
            return

    summary = []
    for i, agent_config in enumerate(agent_configs, start=1):
        register_env("env", lambda _, ac=agent_config: GridWorldEnv(
            size=GRID_SIZE,
            num_lava_tiles=NUM_LAVA_TILES,
            single_agent=False,
            max_steps=MAX_ENV_STEPS,
            proposer_sees_lava=ac.proposer_sees_lava,
            # Exploring starts are PER-ROLE (eval always uses the fixed start):
            #   validator-learning -> ON: the veto decision is local/start-independent,
            #     and random starts expose more forward-into-lava cases (denser signal),
            #     with no train/eval mismatch.
            #   proposer-learning  -> OFF: the proposer must specialize to the FIXED
            #     eval start; random-start training spreads a weak learner's capacity and
            #     tanks fixed-start eval (dropped the SAC proposer from ~89% to ~37%).
            randomize_spawn=(ac.validator_policy == ValidatorPolicies.LEARNED),
        ))
        print(f"\n{'=' * 78}")
        print(f">>> [{i}/{len(agent_configs)}] Training: {experiment_name(agent_config)}")
        print(f"{'=' * 78}\n")
        summary.append(train_one(agent_config, args.iters, args.samples))

    # make sure each run only writes to the experiments it has run
    # so that it does not mess up the merges
    if args.only:
        suffix = "_" + "_".join(s.strip() for s in args.only.split(",") if s.strip())
    else:
        suffix = ""
    out = LOG_DIR / "tune" / f"all_experiments_summary{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Merge with any existing summary
    merged_by_name = {}
    if out.exists():
        try:
            for entry in json.loads(out.read_text()):
                merged_by_name[entry["experiment"]] = entry
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"warning: existing summary at {out} is unreadable; overwriting")
    for entry in summary:
        merged_by_name[entry["experiment"]] = entry
    merged = list(merged_by_name.values())

    out.write_text(json.dumps(merged, indent=2))

    print("\n" + "=" * 78)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 78)
    print(f"\nFull summary (final params + search space + final metrics) saved to:\n  {out}\n")
    print("Compact overview (all experiments, including any previously saved):")
    for entry in merged:
        m = entry.get("final_metrics") or {}
        # episode_return_mean is nested under env_runners/ in the new RLlib API stack
        env_runners = m.get("env_runners", {}) if isinstance(m, dict) else {}
        ret = env_runners.get("episode_return_mean") if isinstance(env_runners, dict) else None
        iters_done = m.get("training_iteration") if isinstance(m, dict) else None
        print(f"  - {entry['experiment']:<80}  iters={iters_done}  return={ret}")

    ray.shutdown()


if __name__ == "__main__":
    main()
