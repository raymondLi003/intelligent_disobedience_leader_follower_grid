"""Evaluate every trained pairing plus the LLM validators.

Usage:
    python run_all_eval.py
    python run_all_eval.py --max-configs 200
"""

import argparse
from pathlib import Path

import ray
from ray.rllib.core.rl_module import RLModule
from ray.tune import register_env

from env import GridWorldEnv
from eval_common import (
    add_config_sampling_args,
    always_approve_factory,
    build_inference_module,
    perfect_proposer_factory,
    perfect_validator_factory,
    print_summary,
    resolve_variations,
    run_pairing,
)
from llm_validator_no_strat import LLMValidatorNoStrat, LLMValidatorNoStratExplain
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from utils import AGENT_CONFIGS, LOG_DIR, ProposerPolicies, ValidatorPolicies, GRID_SIZE, NUM_LAVA_TILES




# LLM validators grouped by provider family 
LLM_FAMILIES = {
    "llama": [
        ("llama4_maverick", "us.meta.llama4-maverick-17b-instruct-v1:0"),
        ("llama4_scout", "us.meta.llama4-scout-17b-instruct-v1:0"),
    ],
    "gpt": [
        ("gpt_5_nano", "gpt-5-nano"),
        ("gpt_5_mini", "gpt-5-mini"),
        ("gpt_5_2", "gpt-5.2"),
    ],
    "gemini": [
        ("gemini_flash_lite", "gemini-2.5-flash-lite"),
        ("gemini_flash", "gemini-2.5-flash"),
        ("gemini_pro", "gemini-2.5-pro"),
    ],
    "claude": [
        ("claude_sonnet45", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("claude_haiku45", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("claude_opus45", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
    ],
}


def select_llm_models(families: list[str]) -> list[tuple[str, str]]:
    """Flatten the requested provider families into (display_name, model_name) pairs."""
    models = []
    for fam in families:
        if fam not in LLM_FAMILIES:
            raise ValueError(f"Unknown LLM family {fam!r}; choose from {sorted(LLM_FAMILIES)}")
        models.extend(LLM_FAMILIES[fam])
    return models

def _make_llm_validator_class(base: type, model_name: str) -> type:
    """Build a subclass of `base` pinned to a specific LLM model."""
    return type(
        f"{base.__name__}_{model_name}",
        (base,),
        {"MODEL_NAME": model_name},
    )

def experiment_name(agent_config) -> str:
    return (
        f"{agent_config.algorithm_name}"
        f"_{agent_config.proposer_policy}_{agent_config.validator_policy}"
        f"__proposer_sees_lava_{agent_config.proposer_sees_lava}"
    )

def load_checkpoint_factory(exp_name: str, policy_id: str):
    def factory(env):
        checkpoint_path = LOG_DIR / "tune" / exp_name / "best_checkpoint" / "learner_group" / "learner" / "rl_module" / policy_id
        if not checkpoint_path.exists():
            # non-RL modules
            if policy_id == ProposerPolicies.PERFECT:
                return perfect_proposer_factory(env)
            if policy_id == ValidatorPolicies.PERFECT:
                return perfect_validator_factory(env)
            if policy_id == ValidatorPolicies.ALWAYS_APPROVE:
                return always_approve_factory(env)
            if policy_id == ProposerPolicies.RANDOM:
                return build_inference_module(env, "proposer", RandomRLModule)
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        return RLModule.from_checkpoint(str(checkpoint_path))
    return factory

def main():
    parser = argparse.ArgumentParser()
    add_config_sampling_args(parser)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--llm-families",
        type=str,
        default="llama,gpt,gemini,claude",
        help="comma-separated provider families to eval (llama,gpt,gemini,claude)",
    )
    parser.add_argument(
        "--llm-explain",
        action="store_true",
    )
    args = parser.parse_args()

    llm_families = [x.strip() for x in args.llm_families.split(",") if x.strip()]
    llm_models = select_llm_models(llm_families)
    llm_base = LLMValidatorNoStratExplain if args.llm_explain else LLMValidatorNoStrat
    llm_suffix = "__explain" if args.llm_explain else ""
    print(f"LLM families {llm_families} -> {len(llm_models)} model(s): {[m[0] for m in llm_models]}"
          f"{' [EXPLAIN mode]' if args.llm_explain else ''}")

    ray.init(ignore_reinit_error=True)
    register_env("env", lambda _: GridWorldEnv(GRID_SIZE, num_lava_tiles=NUM_LAVA_TILES, single_agent=False))

    variations = resolve_variations(args, tag="run_all_eval")

    pairings = []

    # all 12 pairings across algos
    for algo in ["dqn", "ppo", "sac"]:
        for ac in AGENT_CONFIGS:
            # override algo for eval
            from dataclasses import replace
            ac_algo = replace(ac, algorithm_name=algo)
            
            exp_name = experiment_name(ac_algo)
            shortcut_name = f"{ac_algo.algorithm_name}_{ac_algo.proposer_policy}_x_{ac_algo.validator_policy}"
            p_factory = load_checkpoint_factory(exp_name, ac_algo.proposer_policy)
            v_factory = load_checkpoint_factory(exp_name, ac_algo.validator_policy)
            pairings.append((shortcut_name, p_factory, v_factory))

    # LLM validators vs perfect (BFS) proposer
    for display_name, model_name in llm_models:
        validator_class = _make_llm_validator_class(llm_base, model_name)
        # bind class as default so each lambda captures the right one
        llm_factory = lambda env, vc=validator_class: build_inference_module(env, "validator", vc)
        pairings.append((
            f"perfect_x_llm_{display_name}{llm_suffix}",
            perfect_proposer_factory,
            llm_factory,
        ))

    if args.only:
        needles = [s.strip() for s in args.only.split(",") if s.strip()]
        pairings = [p for p in pairings if any(n in p[0] for n in needles)]
        print(f"--only matched {len(pairings)} pairing(s): {[p[0] for p in pairings]}")
        if not pairings:
            print("Nothing to run. Exiting.")
            ray.shutdown()
            return

    results = []

    for name, p_factory, v_factory in pairings:
        print(f"\n>>> Running evaluation for: {name}")

        # per-model dir so videos don't overwrite each other
        if name.startswith("perfect_x_llm_"):
            model_slug = name[len("perfect_x_llm_"):]
            folder = f"videos/llm/{model_slug}"
        else:
            folder = f"videos/empirical/{name.split('_')[0]}"
        
        try:
            res = run_pairing(
                name=name,
                proposer_factory=p_factory,
                validator_factory=v_factory,
                variations=variations,
                video_dir=folder,
                save_video=True,
            )
            
            print(f"\nEvaluating on {GRID_SIZE}x{GRID_SIZE} grid with {NUM_LAVA_TILES} lava tiles")
            print(f"with {name} policies.")
            print(" Proposer ".center(50, '='))
            print(f"Reached the goal in {res['goal_wins']} out of {res['n_configs']} "
                  f"({res['goal_pct']:.2f}%).")

            print(" Validator ".center(50, '='))
            print(f"Validator final rewards mean: {res['validator_mean_reward']}")
            print(f"Validator wanted behaviour: {res['wanted_pct']:.2f}%")
            
            total_disobey = res['total_disobey']
            print(f"Validator total disobediences: {total_disobey} "
                  f"out of {res['n_validator_decisions']} decisions")
            print("Of these,")
            print(f"Validator good disobedience: {res['good_disobey']} "
                  f"({res['good_disobey_rel_pct']:.2f}% of disobediences)")
            print(f"Validator bad disobedience: {res['bad_disobey']} "
                  f"({res['bad_disobey_rel_pct']:.2f}% of disobediences)")

            results.append(res)
        except FileNotFoundError as e:
            print(f"[Skipping {name}] {e}. (Did you train this model first?)")

    if results:
        print_summary(results, len(variations))
    else:
        print("No evaluations were run.")

    ray.shutdown()

if __name__ == '__main__':
    main()
