# The Intelligent Disobedience Game: Leader-Follower Grid

**When should an AI agent disobey the human it's helping in order to keep them safe?**

This repo is a small grid world for studying that question. It accompanies an ongoing research
project on intelligent disobedience.


## The idea

Think of a guide dog that refuses to walk its handler into traffic. The handler gives the command,
but the dog can see a danger the handler can't. As the result, the dog disobeys on purpose to help.

We call this **intelligent disobedience**, and we model it as a game between two agents:

- A **proposer** (the leader) suggests the next move toward a goal.
- A **validator** (the follower) either obeys the move or blocks it.

The twist is **information asymmetry**: the validator can sometimes see hazards (lava tiles) that the
proposer can't. A good validator learns to obey most of the time but step in when obeying would cause harm.
Some maps also have **safety traps**, where a move looks fine now but leads to trouble a few steps later.

## What's in here

The environment and the agents:

- `env.py` — the grid world. A proposer suggests a direction, a validator approves or blocks it, and
  the environment resolves what actually happens.
- `rl_modules/` — the agents. This includes the learnable RL policies plus a few hand-written baselines:
  a **perfect proposer** (always finds a safe path), a **perfect validator** (always makes the right
  obey/disobey call), and an **always-approve validator** (never disobeys).
- `config.py`, `utils.py` — settings and shared helpers (grid size, number of lava tiles, model configs, etc.).
- `metrics.py` — logging callbacks used during training.

The scripts you run:

- `run_all_experiments.py` — train every combination: 4 proposer/validator pairings × 3 algorithms
  (DQN, PPO, SAC), 12 models in total.
- `run_seeds.py` — retrain the chosen configs across several random seeds and report mean ± std.
- `run_all_eval.py` — evaluate the trained models (and an LLM-based validator) on a fixed set of maps,
  print result tables, and save rollout videos.
- `eval_common.py` — the shared rollout / metrics / table code that the eval scripts build on.

Tests live in `tests/`.

## Setup

```bash
pip install -r requirements.txt
```

Most knobs (grid size, number of lava tiles, training iterations) live in `config.py` and `utils.py`.

## Running it

Train everything:

```bash
python run_all_experiments.py
```

```bash
python run_all_experiments.py --iters 200     # fewer iterations
python run_all_experiments.py --samples 16    # turn on autotune with 16 samples
```

Retrain the picked configs across seeds:

```bash
python run_seeds.py
```

Evaluate the trained models:

```bash
python run_all_eval.py
```

Run the tests:

```bash
python -m pytest
```

## Where the output goes

A training run writes into a few folders (all git-ignored):

- `logs/tune/` — one folder per pairing/algorithm, holding the Ray Tune trials, their checkpoints,
  per-iteration metrics, and a `best_checkpoint/` with the winning model. Each algorithm also gets an
  `all_experiments_summary_<algo>.json` summarizing the best trial per pairing.
- `eval_results/` — one text file per pairing with goal rate, validator behavior, and other metrics.
- `videos/` — rollout videos from evaluation.


