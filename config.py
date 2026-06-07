from functools import partial
from typing import Hashable

from ray import tune
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule

from rl_modules.always_approve_validator import AlwaysApproveValidatorRLM
from rl_modules.perfect_proposer import PerfectProposerRLM
from rl_modules.perfect_validator import PerfectValidatorRLM
from utils import (
    AgentConfig,
    ProposerPolicies,
    ValidatorPolicies,
    PROPOSER_ALGORITHM_MODULES,
    VALIDATOR_ALGORITHM_MODULES,
    SINGLE_AGENT_ALGORITHM_MODULES,
    DEFAULT_MULTI_AGENT_MODEL_CONFIG,
    DEFAULT_SINGLE_AGENT_CONV_MODEL_CONFIG, CATALOG_CLASS, )


def create_algorithm_config(algorithm_name: str) -> AlgorithmConfig:
    config = None
    if algorithm_name == "dqn":
        config = DQNConfig().training(
            replay_buffer_config={
                "enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "capacity": 100_000,
                "alpha": 0.8,
                "beta": 0.4,
            },
            train_batch_size_per_learner=2048,
            num_steps_sampled_before_learning_starts=300,
        )

    if algorithm_name == "ppo":
        config = PPOConfig().training(
            entropy_coeff=0.01,
            train_batch_size=1024,
        )

    if algorithm_name == "sac":
        config = SACConfig().training(
            replay_buffer_config={
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "capacity": 100_000,
                "alpha": 0.8,
                "beta": 0.4,
            },
            train_batch_size_per_learner=2048,
            num_steps_sampled_before_learning_starts=300,
            initial_alpha=0.2,
        )

    if config is None:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    return config.framework("torch")


def get_search_space(algorithm_name: str, learns_validator: bool = False) -> dict:
    """Hyperparameter search space for Ray Tune autotune.
    """
    n_step_choices = [1, 3] if learns_validator else [1, 3, 5]
    if algorithm_name == "dqn":
        return {
            "lr": tune.loguniform(1e-5, 3e-4),
            "gamma": tune.uniform(0.95, 0.999),
            "target_network_update_freq": tune.choice([200, 500, 1000]),
            "n_step": tune.choice(n_step_choices),
            # epsilon-greedy schedule
            "epsilon": tune.choice([
                [[0, 1.0], [50_000, 0.05]],
                [[0, 1.0], [150_000, 0.05]],
                [[0, 1.0], [400_000, 0.10]],
            ]),
        }
    if algorithm_name == "ppo":
        return {
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.0, 0.05),
            "clip_param": tune.uniform(0.1, 0.4),
            "num_epochs": tune.choice([10, 20, 30]),
        }
    if algorithm_name == "sac":
        return {
            "actor_lr": tune.loguniform(1e-5, 1e-3),
            "critic_lr": tune.loguniform(1e-5, 1e-3),
            "tau": tune.uniform(0.001, 0.01),
            "gamma": tune.uniform(0.95, 0.999),
            "n_step": tune.choice(n_step_choices),
        }
    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def add_env_config(config: AlgorithmConfig) -> AlgorithmConfig:
    config.environment("env")
    # needed for turn-based env
    # num env runner is for cpu parallelism 
    config.env_runners(
        batch_mode="complete_episodes",
        num_env_runners=6,
        num_cpus_per_env_runner=1,
    )
    return config


def add_single_agent_policies(config: AlgorithmConfig, agent_config: AgentConfig) -> AlgorithmConfig:
    module_class = SINGLE_AGENT_ALGORITHM_MODULES[agent_config.algorithm_name]

    return config.multi_agent(
        policies=["single_agent"],
        policy_mapping_fn=lambda agent_id, episode: "single_agent",
        policies_to_train=["single_agent"],
    ).rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "single_agent": RLModuleSpec(
                    module_class=module_class,
                    model_config=DEFAULT_SINGLE_AGENT_CONV_MODEL_CONFIG,
                )
            }
        )
    )


def agent_config_policy_mapping(
        agent_id: Hashable,
        episode: MultiAgentEpisode,
        agent_config: AgentConfig,
) -> str:
    if agent_id == "proposer":
        return agent_config.proposer_policy

    if agent_id == "validator":
        return agent_config.validator_policy

    raise ValueError(f"Invalid agent: {agent_id}")


def get_multi_agent_rl_module_specs(policy_names: list[str], agent_config: AgentConfig) -> dict[str, RLModuleSpec]:
    rl_module_specs = {}
    if ProposerPolicies.LEARNED in policy_names:
        rl_module_specs[ProposerPolicies.LEARNED] = RLModuleSpec(
            module_class=PROPOSER_ALGORITHM_MODULES[agent_config.algorithm_name],
            model_config=DEFAULT_MULTI_AGENT_MODEL_CONFIG,
            catalog_class=CATALOG_CLASS[agent_config.algorithm_name],
        )

    if ProposerPolicies.PERFECT in policy_names:
        rl_module_specs[ProposerPolicies.PERFECT] = RLModuleSpec(
            module_class=PerfectProposerRLM,
            inference_only=True,
        )

    if ProposerPolicies.RANDOM in policy_names:
        rl_module_specs[ProposerPolicies.RANDOM] = RLModuleSpec(
            module_class=RandomRLModule,
            inference_only=True,
        )

    if ValidatorPolicies.LEARNED in policy_names:
        rl_module_specs[ValidatorPolicies.LEARNED] = RLModuleSpec(
            module_class=VALIDATOR_ALGORITHM_MODULES[agent_config.algorithm_name],
            model_config=DEFAULT_MULTI_AGENT_MODEL_CONFIG,
            catalog_class=CATALOG_CLASS[agent_config.algorithm_name],
        )

    if ValidatorPolicies.PERFECT in policy_names:
        rl_module_specs[ValidatorPolicies.PERFECT] = RLModuleSpec(
            module_class=PerfectValidatorRLM,
            inference_only=True,
        )

    if ValidatorPolicies.ALWAYS_APPROVE in policy_names:
        rl_module_specs[ValidatorPolicies.ALWAYS_APPROVE] = RLModuleSpec(
            module_class=AlwaysApproveValidatorRLM,
            inference_only=True,
        )

    return rl_module_specs


def add_multi_agent_policies(
        config: AlgorithmConfig,
        agent_config: AgentConfig,
) -> AlgorithmConfig:
    assert agent_config.proposer_policy is not None
    assert agent_config.validator_policy is not None

    policies = [agent_config.proposer_policy, agent_config.validator_policy]
    policy_mapping_fn = partial(agent_config_policy_mapping, agent_config=agent_config)
    policies_to_train = []
    if agent_config.proposer_policy == ProposerPolicies.LEARNED:
        policies_to_train.append(ProposerPolicies.LEARNED)
    if agent_config.validator_policy == ValidatorPolicies.LEARNED:
        policies_to_train.append(ValidatorPolicies.LEARNED)

    config = (
        config
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=get_multi_agent_rl_module_specs(policies, agent_config),
            )
        )
    )

    return config


def create_rllib_config(agent_config: AgentConfig) -> AlgorithmConfig:
    config = create_algorithm_config(agent_config.algorithm_name)
    config = add_env_config(config)
    if agent_config.proposer_policy is None and agent_config.validator_policy is None:
        assert agent_config.algorithm_name in SINGLE_AGENT_ALGORITHM_MODULES.keys()
        config = add_single_agent_policies(config, agent_config)
    else:
        assert agent_config.algorithm_name in PROPOSER_ALGORITHM_MODULES.keys()
        assert agent_config.algorithm_name in VALIDATOR_ALGORITHM_MODULES.keys()
        config = add_multi_agent_policies(config, agent_config)

    config.validate()

    return config
