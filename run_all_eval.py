"""Consolidated evaluation script for all IDG experiments.

Evaluates the 12 empirical pairings (DQN, PPO, SAC) along with 
the LLM-validator

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
from llm_validator_no_strat import (
    LLMValidatorNoStrat,
    LLMValidatorNoStratRulebook,
)
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from utils import AGENT_CONFIGS, LOG_DIR, ProposerPolicies, ValidatorPolicies, GRID_SIZE, NUM_LAVA_TILES



# level 1 to level 3 in terms of model advance level
LLM_LEVELS = {
    1: [
        ("l1_claude_haiku3", "us.anthropic.claude-3-haiku-20240307-v1:0"),
        ("l1_gpt_4o_mini", "4o-mini"),
        ("l1_gemini_flash_lite", "gemini-2.5-flash-lite"),
    ],
    2: [
        ("l2_claude_haiku45", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("l2_gpt_5_mini", "gpt-5-mini"),
        ("l2_gemini_flash", "gemini-2.5-flash"),
    ],
    3: [
        ("l3_claude_opus45", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
        ("l3_gpt_5_2", "gpt-5.2"),
        ("l3_gemini_pro", "gemini-2.5-pro"),
    ],
}


def select_llm_models(levels: list[int]) -> list[tuple[str, str]]:
    """Flatten the requested complexity levels into (display_name, model_name) pairs."""
    models = []
    for lvl in levels:
        if lvl not in LLM_LEVELS:
            raise ValueError(f"Unknown LLM level {lvl}; choose from {sorted(LLM_LEVELS)}")
        models.extend(LLM_LEVELS[lvl])
    return models

LLM_VARIANTS = [
    ("", LLMValidatorNoStrat),
    ("__rulebook", LLMValidatorNoStratRulebook),
]


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
            # create inference modules for non-RL modules
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
        help="comma-separated substrings; only run pairings whose name matches one. "
             "e.g. --only ppo_perfect_proposer_x_learned_validator",
    )
    parser.add_argument(
        "--llm-levels",
        type=str,
        default="1,2,3",
        help="comma-separated LLM complexity levels to eval (1=rudimentary, 2=mid, "
             "3=frontier). e.g. --llm-levels 1 or --llm-levels 1,3",
    )
    args = parser.parse_args()

    llm_levels = [int(x) for x in args.llm_levels.split(",") if x.strip()]
    llm_models = select_llm_models(llm_levels)
    print(f"LLM levels {llm_levels} -> {len(llm_models)} model(s): {[m[0] for m in llm_models]}")

    ray.init(ignore_reinit_error=True)
    register_env("env", lambda _: GridWorldEnv(GRID_SIZE, num_lava_tiles=NUM_LAVA_TILES, single_agent=False))

    variations = resolve_variations(args, tag="run_all_eval")

    pairings = []

    # Get all 12 experiment pairings automatically across algos
    for algo in ["dqn", "ppo", "sac"]:
        for ac in AGENT_CONFIGS:
            # override the original agent config for eval
            from dataclasses import replace
            ac_algo = replace(ac, algorithm_name=algo)
            
            exp_name = experiment_name(ac_algo)
            shortcut_name = f"{ac_algo.algorithm_name}_{ac_algo.proposer_policy}_x_{ac_algo.validator_policy}"
            p_factory = load_checkpoint_factory(exp_name, ac_algo.proposer_policy)
            v_factory = load_checkpoint_factory(exp_name, ac_algo.validator_policy)
            pairings.append((shortcut_name, p_factory, v_factory))

    # LLM validators paired with the perfect (BFS) proposer
    for variant_suffix, base_cls in LLM_VARIANTS:
        for display_name, model_name in llm_models:
            validator_class = _make_llm_validator_class(base_cls, model_name)
            # bind vars as defaults so each lambda closure captures the right class
            llm_factory = lambda env, vc=validator_class: build_inference_module(env, "validator", vc)
            pairings.append((
                f"perfect_x_llm_{display_name}{variant_suffix}",
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
    
    # Use output folder mapping similar to run_eval_no_llm 
    for name, p_factory, v_factory in pairings:
        print(f"\n>>> Running evaluation for: {name}")
        
        # Decide out folder logic. LLM runs get their own per-model dir so
        # videos from different models don't overwrite each other
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
            
            # line by line eval
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
