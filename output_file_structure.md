```text
intelligent_disobedience_results_autotune/
├── logs/                                # master training outputs
│   └── tune/
│       ├── dqn_random_proposer_learned_validator__proposer_sees_lava_False/   # one folder per pairing for three algos,
│       │                                                                      # DQN, PPO, and SAC. 4 experiments for
│       │                                                                      # each algo. 12 in total.
│       │   ├── DQN_env_xxxxx_00000_..._timestamp/    # one Ray Tune trial folder per sampled config
│       │   │   ├── checkpoint_0000NN/                # periodic model checkpoint thats saved every 50 iterations
│       │   │   ├── params.json                       # the trial's hyperparameters
│       │   │   ├── result.json                       # per-iteration metrics, one JSON record per line
│       │   │   ├── progress.csv                      # same metrics as CSV
│       │   │   └── events.out.tfevents.*             # TensorBoard log
│       │   ├── best_checkpoint/                      # the best trial's final weights (we uses this for eval)
│       │   ├── experiment_state-*.json               # Ray Tune bookkeeping
│       │   └── tuner.pkl
│       ├── all_experiments_summary_dqn.json          # summary of best trial per pairing, one file per algorithm
│       ├── all_experiments_summary_ppo.json
│       └── all_experiments_summary_sac.json
├── eval_results_20260530-1443/          # one text file per pairing with goal rate, validator behaviour,
│                                        # and other metrics
└── videos_20260530-1443/                # eval rollout videos
```