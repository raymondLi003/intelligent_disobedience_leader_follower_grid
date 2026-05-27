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
"""

import argparse
import json
from pathlib import Path

import ray
from ray import tune
from ray.tune import Checkpoint, register_env

from config import create_rllib_config
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


def train_one(agent_config: AgentConfig, iters: int) -> dict:
    config = create_rllib_config(agent_config)
    config.callbacks([ActionLoggerCallback])

    name = experiment_name(agent_config)
    param_space = config.to_dict()

    tuner = tune.Tuner(
        config.algo_class,
        param_space=param_space,
        run_config=tune.RunConfig(
            stop={"training_iteration": iters},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=10,
            ),
            storage_path=LOG_DIR / "tune",
            name=name,
            callbacks=[CustomTBXLoggerCallback()],
        ),
    )

    results = tuner.fit()
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
        help=f"training iterations per experiment (default {TRAINING_ITERATIONS})",
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

    summary = []
    for i, agent_config in enumerate(agent_configs, start=1):
        register_env("env", lambda _, ac=agent_config: GridWorldEnv(
            size=GRID_SIZE,
            num_lava_tiles=NUM_LAVA_TILES,
            single_agent=False,
            max_steps=MAX_ENV_STEPS,
            proposer_sees_lava=ac.proposer_sees_lava,
        ))
        print(f"\n{'=' * 78}")
        print(f">>> [{i}/{len(agent_configs)}] Training: {experiment_name(agent_config)}")
        print(f"{'=' * 78}\n")
        summary.append(train_one(agent_config, args.iters))

    out = LOG_DIR / "tune" / "all_experiments_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 78)
    print(f"\nFull summary (final params + search space + final metrics) saved to:\n  {out}\n")
    print("Compact overview:")
    for entry in summary:
        m = entry.get("final_metrics") or {}
        # episode_return_mean is nested under env_runners/ in the new RLlib API stack
        env_runners = m.get("env_runners", {}) if isinstance(m, dict) else {}
        ret = env_runners.get("episode_return_mean") if isinstance(env_runners, dict) else None
        iters_done = m.get("training_iteration") if isinstance(m, dict) else None
        print(f"  - {entry['experiment']:<80}  iters={iters_done}  return={ret}")

    ray.shutdown()


if __name__ == "__main__":
    main()
