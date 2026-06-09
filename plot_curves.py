"""Plot learning curves 
Usage:
    python plot_curves.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
TUNE = os.path.join(REPO, "logs", "tune")
OUT = os.path.join(REPO, "curves")
os.makedirs(OUT, exist_ok=True)

ITER = "training_iteration"


def metric_for(name):
    if "learned_proposer" in name:
        return "env_runners/module_episode_returns_mean/learned_proposer"
    if "learned_validator" in name:
        return "env_runners/module_episode_returns_mean/learned_validator"
    return "env_runners/episode_return_mean"


def trial_curves(exp_dir, metric):
    """yield (trial_name, DataFrame[iter, metric]) for every trial that has the metric."""
    for csv in sorted(glob.glob(os.path.join(exp_dir, "*", "progress.csv"))):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if metric not in df.columns or ITER not in df.columns:
            continue
        s = df[[ITER, metric]].dropna()
        if not s.empty:
            yield os.path.basename(os.path.dirname(csv)), s


def plot_panel(ax, exp_dir, metric, title):
    curves = list(trial_curves(exp_dir, metric))
    max_len = max((s[ITER].iloc[-1] for _, s in curves), default=0)
    best = None
    best_final = -1e18
    for _, s in curves:
        ax.plot(s[ITER], s[metric], color="0.8", lw=0.6, zorder=1)
        if s[ITER].iloc[-1] >= max_len and s[metric].iloc[-1] > best_final:
            best_final = s[metric].iloc[-1]
            best = s
    if best is not None:
        ax.plot(best[ITER], best[metric], color="C0", lw=1.8, zorder=2, label="best full-length trial")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return best is not None


ALGOS = ["dqn", "ppo", "sac"]
ROLES = [("proposer", "learned_proposer_perfect_validator"),
         ("validator", "perfect_proposer_learned_validator")]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for col, algo in enumerate(ALGOS):
    for row, (role, suffix) in enumerate(ROLES):
        name = f"{algo}_{suffix}__proposer_sees_lava_False"
        d = os.path.join(TUNE, name)
        ax = axes[row][col]
        if not os.path.isdir(d):
            ax.set_title(f"{algo.upper()} {role} (missing)", fontsize=10)
            continue
        m = metric_for(name)
        ok = plot_panel(ax, d, m, f"{algo.upper()} {role}")
        if col == 0:
            ax.set_ylabel("learned-side return")
        if row == 1:
            ax.set_xlabel("training iteration")
        # also save the individual panel as its own figure
        if ok:
            f1, a1 = plt.subplots(figsize=(6, 4))
            plot_panel(a1, d, m, f"{algo.upper()} {role}  ({name})")
            a1.set_xlabel("training iteration"); a1.set_ylabel(m.split("/")[-1])
            f1.tight_layout(); f1.savefig(os.path.join(OUT, f"{name}.png"), dpi=120)
            plt.close(f1)

fig.suptitle("learning curves (grey - all trials, blue - the best checkpoint)")
fig.tight_layout()
grid_path = os.path.join(OUT, "learning_curves.png")
fig.savefig(grid_path, dpi=120)
print(f"wrote {grid_path}")
print(f"plus per-pairing PNGs in {OUT}/")
