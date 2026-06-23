import copy
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



DQN_PROPOSER_MODEL_CONFIG = copy.deepcopy(DEFAULT_MULTI_AGENT_MODEL_CONFIG)
DQN_PROPOSER_MODEL_CONFIG["head_fcnet_hiddens"] = [256, 128]


def _prioritized_episode_buffer(capacity: int, alpha: float, beta: float = 0.4,
                                enable_api: bool = False) -> dict:
    """Prioritized multi-agent episode replay buffer config shared by DQN and SAC."""
    cfg = {}
    if enable_api:
        cfg["enable_replay_buffer_api"] = True
    cfg.update({
        "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
        "capacity": capacity,
        "alpha": alpha,
        "beta": beta,
    })
    return cfg


def _dqn_algorithm_config(learns_validator: bool) -> AlgorithmConfig:
    if learns_validator:
        epsilon = [(0, 1.0), (10_000, 0.05)]
        buffer_capacity = 500_000
        replay_alpha = 0.5
        warmup = 2_000
    else:
        epsilon = [(0, 1.0), (10_000, 0.05)]
        buffer_capacity = 500_000
        replay_alpha = 0.5
        warmup = 2_000
    return DQNConfig().training(
        replay_buffer_config=_prioritized_episode_buffer(
            buffer_capacity, replay_alpha, enable_api=True),
        train_batch_size_per_learner=512,
        num_steps_sampled_before_learning_starts=warmup,
        epsilon=epsilon,
        n_step=1,
    )


def _ppo_algorithm_config(learns_validator: bool) -> AlgorithmConfig:
    if learns_validator:
        entropy_coeff = 0.01
    else:
        entropy_coeff = [
            (0, 0.2),
            (200_000, 0.05),
            (800_000, 0.005),
        ]
    return PPOConfig().training(entropy_coeff=entropy_coeff, train_batch_size=512)


def _sac_algorithm_config(learns_validator: bool) -> AlgorithmConfig:
    if learns_validator:
        target_entropy = "auto"
        n_step = 1
    else:
        target_entropy = "auto"
        n_step = 1
    return SACConfig().training(
        replay_buffer_config=_prioritized_episode_buffer(100_000, 0.8),
        train_batch_size_per_learner=512,
        num_steps_sampled_before_learning_starts=300,
        initial_alpha=0.2,
        target_entropy=target_entropy,
        n_step=n_step,
    )


_ALGORITHM_CONFIG_BUILDERS = {
    "dqn": _dqn_algorithm_config,
    "ppo": _ppo_algorithm_config,
    "sac": _sac_algorithm_config,
}


def create_algorithm_config(algorithm_name: str, learns_validator: bool = False) -> AlgorithmConfig:
    try:
        builder = _ALGORITHM_CONFIG_BUILDERS[algorithm_name]
    except KeyError:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return builder(learns_validator).framework("torch")


def _dqn_search_space(learns_validator: bool) -> dict:
    if learns_validator:
        return {
            "lr": tune.loguniform(3e-5, 1.5e-4),
            "gamma": tune.uniform(0.92, 0.965),
            "target_network_update_freq": tune.choice([200, 500]),
            "n_step": tune.choice([1, 3]),
            "epsilon": tune.choice([
                [(0, 1.0), (80_000, 0.10), (200_000, 0.02)],
                [(0, 1.0), (120_000, 0.10), (230_000, 0.02)],
                [(0, 1.0), (150_000, 0.15), (240_000, 0.03)],
            ]),
        }
    return {
        "lr": tune.loguniform(3e-5, 1.5e-4),
        "gamma": tune.uniform(0.92, 0.965),
        "target_network_update_freq": tune.choice([200, 500]),
        "n_step": tune.choice([2, 3, 4]),
        "epsilon": tune.choice([
            [(0, 1.0), (200_000, 0.20), (500_000, 0.10)],
            [(0, 1.0), (300_000, 0.20), (650_000, 0.05)],
            [(0, 1.0), (250_000, 0.15), (550_000, 0.10)],
        ]),
        "replay_buffer_config": tune.choice([
            _prioritized_episode_buffer(100_000, 0.8, enable_api=True),
            _prioritized_episode_buffer(250_000, 0.8, enable_api=True),
        ]),
    }


def _ppo_search_space(learns_validator: bool) -> dict:
    if learns_validator:
        return {
            "lr": tune.loguniform(2.5e-4, 3.5e-4),
            "gamma": tune.uniform(0.94, 0.955),
            "entropy_coeff": tune.uniform(0.004, 0.01),
            "clip_param": tune.uniform(0.14, 0.2),
            "num_epochs": tune.choice([10, 15]),
        }
    return {
        "lr": tune.loguniform(1e-5, 1e-3),
        "entropy_coeff": tune.uniform(0.05, 0.25),
        "clip_param": tune.uniform(0.1, 0.4),
        "num_epochs": tune.choice([10, 20, 30]),
    }


def _sac_search_space(learns_validator: bool) -> dict:
    if learns_validator:
        return {
            "actor_lr": tune.loguniform(1e-5, 1e-3),
            "critic_lr": tune.loguniform(1e-5, 1e-3),
            "alpha_lr": tune.loguniform(3e-4, 1e-2),
            "tau": tune.uniform(0.001, 0.01),
            "gamma": tune.uniform(0.95, 0.999),
            "n_step": tune.choice([1, 3]),
            "initial_alpha": tune.uniform(0.2, 0.8),
        }
    return {
        "actor_lr": tune.loguniform(1e-5, 1e-3),
        "critic_lr": tune.loguniform(1e-5, 1e-3),
        "alpha_lr": tune.loguniform(3e-4, 1e-2),
        "tau": tune.uniform(0.001, 0.01),
        "gamma": tune.uniform(0.99, 0.997),
        "n_step": tune.choice([2, 3, 4]),
        "initial_alpha": tune.uniform(0.1, 0.3),
        "target_entropy": tune.uniform(0.2, 0.5),
        "replay_buffer_config": tune.choice([
            _prioritized_episode_buffer(100_000, 0.8),
            _prioritized_episode_buffer(250_000, 0.8),
        ]),
    }


_SEARCH_SPACE_BUILDERS = {
    "dqn": _dqn_search_space,
    "ppo": _ppo_search_space,
    "sac": _sac_search_space,
}


def get_search_space(algorithm_name: str, learns_validator: bool = False) -> dict:
    """Hyperparameter search space for Ray Tune autotune."""
    try:
        builder = _SEARCH_SPACE_BUILDERS[algorithm_name]
    except KeyError:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return builder(learns_validator)


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
        proposer_model_config = (
            DQN_PROPOSER_MODEL_CONFIG
            if agent_config.algorithm_name == "dqn"
            else DEFAULT_MULTI_AGENT_MODEL_CONFIG
        )
        rl_module_specs[ProposerPolicies.LEARNED] = RLModuleSpec(
            module_class=PROPOSER_ALGORITHM_MODULES[agent_config.algorithm_name],
            model_config=proposer_model_config,
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
    learns_validator = agent_config.validator_policy == ValidatorPolicies.LEARNED
    config = create_algorithm_config(agent_config.algorithm_name, learns_validator=learns_validator)
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
