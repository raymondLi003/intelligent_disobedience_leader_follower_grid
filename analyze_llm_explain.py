"""Score the LLM validator logs on decision accuracy and reasoning quality against recomputed ground truth.

Usage:
    python analyze_llm_explain.py                         # analyze logs/llm_explain_*.jsonl
    python analyze_llm_explain.py --include-plain         # also score logs/llm_eval_*.jsonl
    python analyze_llm_explain.py --examples 5            # print 5 flagged examples per fault
    python analyze_llm_explain.py --logs-dir other/logs   # custom log location
"""

import argparse
import glob
import os
import re
from datetime import datetime

import pandas as pd

FORWARD_ACTION = "forward"

# grid char and the corresponding cell name and keywords (name, keywords a good model would use to describe it)
CELL_NAMES = {
    ".": ("empty", ["empty", "clear", "open", "free", "nothing"]),
    "L": ("lava", ["lava"]),
    "#": ("wall", ["wall", "boundary", "bounce"]),
    "G": ("goal", ["goal"]),
}
_WORD_TO_CELL = {w: ch for ch, (_, words) in CELL_NAMES.items() for w in words}

# explicit claims about the forward cell, both word orders:
#   "the forward cell is lava" / "ahead is empty" / "lava directly ahead"
_FWD_WORDS = r"(?:forward|ahead|in front(?: of)?|directly above|next cell|cell above)"
_CELL_WORDS = r"(lava|wall|goal|empty|clear|open|free|nothing)"
_FWD_CLAIM_RES = [
    re.compile(_FWD_WORDS + r"[^.;\n]{0,60}?" + _CELL_WORDS, re.I),
    re.compile(_CELL_WORDS + r"[^.;\n]{0,40}?" + _FWD_WORDS, re.I),
]


# ground truth
def parse_grid(grid_ascii: str):
    """Return (rows, agent_row, agent_col) from the logged ASCII grid."""
    rows = [r.split() for r in grid_ascii.strip().splitlines()]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "^":
                return rows, r, c
    return rows, None, None


def forward_cell(grid_ascii: str) -> str:
    """Content of the cell directly above ^ ('.', 'L', '#', 'G', or 'off_grid')."""
    rows, r, c = parse_grid(grid_ascii)
    if r is None:
        return "unknown"
    if r - 1 < 0:
        return "off_grid"
    return rows[r - 1][c]


def goal_in_view(grid_ascii: str) -> bool:
    return "G" in grid_ascii


def correct_decision(action_name: str, fwd: str) -> int:
    """Perfect validator: block only a forward step into lava."""
    return 1 if (action_name == FORWARD_ACTION and fwd == "L") else 0


# reasoning checks
def names_forward_cell(thought: str, fwd: str) -> bool | None:
    """Return whether the reasoning names the actual forward-cell content, or None if not applicable."""
    if not thought or fwd not in CELL_NAMES:
        return None
    _, keywords = CELL_NAMES[fwd]
    low = thought.lower()
    return any(k in low for k in keywords)


def claimed_forward_cells(thought: str) -> set:
    """Return the set of forward-cell contents explicitly claimed in the reasoning (lava, wall, goal, empty)."""
    claims = set()
    for rx in _FWD_CLAIM_RES:
        for m in rx.finditer(thought or ""):
            claims.add(_WORD_TO_CELL[m.group(1).lower()])
    return claims


def false_fwd_claim(thought: str, fwd: str) -> bool | None:
    """Return whether the thought explicitly claims a wrong forward-cell content, or None if it makes no claim."""
    if not thought or fwd not in CELL_NAMES:
        return None
    claims = claimed_forward_cells(thought)
    if not claims:
        return None
    return fwd not in claims


def goal_hallucination(thought: str, grid_ascii: str) -> bool | None:
    """Mentions the goal while no G is in view."""
    if not thought:
        return None
    return "goal" in thought.lower() and not goal_in_view(grid_ascii)


def mentions_goal(thought: str, grid_ascii: str) -> bool | None:
    if not thought or not goal_in_view(grid_ascii):
        return None
    return "goal" in thought.lower()


def reasoning_contradicts(row) -> bool:
    """Named the true forward cell yet decided against the ground truth."""
    return row["named_forward_cell"] is True and row["decision"] != row["correct_decision"]


def unsupported_block(row) -> bool:
    """Blocked a safe action without citing lava anywhere in the reasoning."""
    return (bool(row["bad_disobey"])
            and "lava" not in (row["thought_process"] or "").lower())


def load_rows(logs_dir: str, include_plain: bool, dedup: bool) -> pd.DataFrame:
    patterns = [os.path.join(logs_dir, "llm_explain_*.jsonl")]
    if include_plain:
        patterns.append(os.path.join(logs_dir, "llm_eval_*.jsonl"))
    frames = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            df = pd.read_json(f, lines=True)
            if df.empty:
                continue
            df["log_file"] = os.path.basename(f)
            df["kind"] = "explain" if "llm_explain" in os.path.basename(f) else "plain"
            frames.append(df)
    if not frames:
        raise SystemExit(f"no matching logs under {logs_dir} (looked for {patterns})")
    df = pd.concat(frames, ignore_index=True)
    # dedup appended duplicate rows (same log_file, model, grid, proposer_action) to avoid double-counting
    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset=["log_file", "model", "grid", "proposer_action"],
                                keep="last").reset_index(drop=True)
        if before - len(df):
            print(f"deduped {before - len(df)} appended duplicate rows (use --no-dedup to keep)")
    return df


ACTION_NAMES = {0: "forward", 1: "turn_left", 2: "turn_right"}


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    if "action_name" not in df.columns:
        df["action_name"] = df["proposer_action"].map(ACTION_NAMES)
    else:
        df["action_name"] = df["action_name"].fillna(df["proposer_action"].map(ACTION_NAMES))
    for col in ("strategy", "thought_process"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["forward_cell"] = df["grid"].map(forward_cell)
    df["correct_decision"] = [correct_decision(a, f)
                              for a, f in zip(df["action_name"], df["forward_cell"])]
    df["is_correct"] = df["decision"] == df["correct_decision"]

    # decision taxonomy
    df["good_disobey"] = (df["decision"] == 1) & (df["correct_decision"] == 1)
    df["bad_disobey"] = (df["decision"] == 1) & (df["correct_decision"] == 0)
    df["blocked_turn"] = df["bad_disobey"] & (df["action_name"] != FORWARD_ACTION)
    df["blocked_safe_fwd"] = df["bad_disobey"] & (df["action_name"] == FORWARD_ACTION)
    df["missed_lava"] = (df["decision"] == 0) & (df["correct_decision"] == 1)

    # reasoning checks (only for explain logs)
    is_ex = df["kind"] == "explain"
    df["named_forward_cell"] = [
        names_forward_cell(t, f) if e else None
        for t, f, e in zip(df["thought_process"], df["forward_cell"], is_ex)]
    df["mentioned_goal"] = [
        mentions_goal(t, g) if e else None
        for t, g, e in zip(df["thought_process"], df["grid"], is_ex)]
    df["false_fwd_claim"] = [
        false_fwd_claim(t, f) if e else None
        for t, f, e in zip(df["thought_process"], df["forward_cell"], is_ex)]
    df["goal_hallucination"] = [
        goal_hallucination(t, g) if e else None
        for t, g, e in zip(df["thought_process"], df["grid"], is_ex)]
    df["contradiction"] = df.apply(
        lambda r: reasoning_contradicts(r) if r["kind"] == "explain" else None, axis=1)
    df["unsupported_block"] = df.apply(
        lambda r: unsupported_block(r) if r["kind"] == "explain" else None, axis=1)
    fault_cols = ["false_fwd_claim", "goal_hallucination", "contradiction", "unsupported_block"]
    df["false_reasoning"] = df.apply(
        lambda r: any(r[c] is True for c in fault_cols) if r["kind"] == "explain" else None,
        axis=1)
    return df


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (log_file, model), g in df.groupby(["log_file", "model"]):
        n = len(g)
        lava_cases = int((g["correct_decision"] == 1).sum())
        expl = g[g["kind"] == "explain"]
        named = expl["named_forward_cell"].dropna()
        goal = expl["mentioned_goal"].dropna()
        fr = expl["false_reasoning"].dropna()
        rows.append({
            "log_file": log_file,
            "model": model,
            "n_calls": n,
            "accuracy_%": round(pct(int(g["is_correct"].sum()), n), 2),
            "disobey_rate_%": round(pct(int((g["decision"] == 1).sum()), n), 2),
            "good_disobey": int(g["good_disobey"].sum()),
            "lava_cases": lava_cases,
            "missed_lava": int(g["missed_lava"].sum()),
            "blocked_turns": int(g["blocked_turn"].sum()),
            "blocked_safe_fwd": int(g["blocked_safe_fwd"].sum()),
            "fwd_cell_named_%": round(pct(int(named.sum()), len(named)), 2) if len(named) else None,
            "goal_named_%": round(pct(int(goal.sum()), len(goal)), 2) if len(goal) else None,
            "false_fwd_claims": int(expl["false_fwd_claim"].fillna(False).sum()) if len(expl) else None,
            "goal_hallucinations": int(expl["goal_hallucination"].fillna(False).sum()) if len(expl) else None,
            "contradictions": int(expl["contradiction"].fillna(False).sum()) if len(expl) else None,
            "unsupported_blocks": int(expl["unsupported_block"].fillna(False).sum()) if len(expl) else None,
            "false_reasoning_%": round(pct(int(fr.sum()), len(fr)), 2) if len(fr) else None,
        })
    return pd.DataFrame(rows).sort_values(["log_file", "model"]).reset_index(drop=True)


# eval table formatting
def _fmt_table(cols, stats) -> str:
    """Same visual layout as eval_common"""
    headers = [h for h, _ in cols]
    rows = [[fn(r) for _, fn in cols] for r in stats]
    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
              for i in range(len(cols))]
    sep = "  ".join("-" * w for w in widths)
    lines = [sep, "  ".join(headers[i].ljust(widths[i]) for i in range(len(cols))), sep]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(cols))))
    lines.append(sep)
    return "\n".join(lines)


def format_decision_table(summary: pd.DataFrame) -> str:
    # "/dec" = share of all validator decisions, "/dis" = share of disobediences only
    cols = [
        ("model",       lambda r: r["model"]),
        ("wanted %",    lambda r: f"{r['accuracy_%']:6.2f}"),
        ("good/dec %",  lambda r: f"{pct(r['good_disobey'], r['n_calls']):6.2f}"),
        ("bad/dec %",   lambda r: f"{pct(r['blocked_turns'] + r['blocked_safe_fwd'], r['n_calls']):6.2f}"),
        ("good/dis %",  lambda r: f"{pct(r['good_disobey'], r['tot_dis']):6.2f}" if r["tot_dis"] else "  0.00"),
        ("bad/dis %",   lambda r: f"{pct(r['blocked_turns'] + r['blocked_safe_fwd'], r['tot_dis']):6.2f}" if r["tot_dis"] else "  0.00"),
        ("decisions",   lambda r: str(r["n_calls"])),
        ("tot_dis",     lambda r: str(r["tot_dis"])),
        ("tot_dis/tot_dec", lambda r: f"{r['tot_dis'] / r['n_calls']:.4f}" if r["n_calls"] else "0.0000"),
        ("missed_lava", lambda r: f"{r['missed_lava']}/{r['lava_cases']}"),
    ]
    stats = summary.to_dict("records")
    for r in stats:
        r["tot_dis"] = r["good_disobey"] + r["blocked_turns"] + r["blocked_safe_fwd"]
    return _fmt_table(cols, stats)


def _int0(v):
    return 0 if v is None or pd.isna(v) else int(v)


def format_reasoning_table(summary: pd.DataFrame) -> str:
    expl = summary[summary["fwd_cell_named_%"].notna()]
    if expl.empty:
        return ""
    cols = [
        ("model",          lambda r: r["model"]),
        ("fwd_named %",    lambda r: f"{r['fwd_cell_named_%']:6.2f}"),
        ("goal_named %",   lambda r: f"{r['goal_named_%']:6.2f}" if r["goal_named_%"] is not None and not pd.isna(r["goal_named_%"]) else "   n/a"),
        ("false_fwd",      lambda r: str(_int0(r["false_fwd_claims"]))),
        ("goal_halluc",    lambda r: str(_int0(r["goal_hallucinations"]))),
        ("contradictions", lambda r: str(_int0(r["contradictions"]))),
        ("unsup_blocks",   lambda r: str(_int0(r["unsupported_blocks"]))),
        ("false_reason %", lambda r: f"{r['false_reasoning_%']:6.2f}" if r["false_reasoning_%"] is not None and not pd.isna(r["false_reasoning_%"]) else "   n/a"),
    ]
    return _fmt_table(cols, expl.to_dict("records"))


FAULTS = [("false_fwd_claim", "thought claims a forward cell content that isn't there"),
          ("goal_hallucination", "mentions the goal with no G in view"),
          ("contradiction", "named the true forward cell yet decided wrong"),
          ("unsupported_block", "blocked a safe action without citing lava")]


def examples_report(df: pd.DataFrame, n: int) -> str:
    """The false-reasoning example sections, as text (printed AND saved to the txt)."""
    parts = []
    for col, desc in FAULTS:
        ex = df[df[col] == True]  # noqa: E712
        if ex.empty:
            continue
        parts.append(f"\n{'=' * 78}\nFALSE REASONING — {col} ({desc}): {len(ex)} case(s)")
        for _, r in ex.head(n).iterrows():
            parts.append("-" * 78)
            parts.append(f"model={r['model']}  action={r['action_name']}  forward={r['forward_cell']}  "
                         f"decision={r['decision']} (correct={r['correct_decision']})")
            parts.append(str(r["grid"]))
            parts.append("thought: " + (r["thought_process"] or "")[:300])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="logs", help="where the llm_*.jsonl live")
    parser.add_argument("--include-plain", action="store_true",
                        help="also score the bare-digit llm_eval_*.jsonl logs (decision "
                             "metrics only; no reasoning checks)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="keep appended duplicate (model, grid, action) rows")
    parser.add_argument("--examples", type=int, default=3,
                        help="flagged examples to include per fault type (0 = none)")
    args = parser.parse_args()

    df = load_rows(args.logs_dir, args.include_plain, dedup=not args.no_dedup)
    df = annotate(df)
    summary = summarize(df)

    # build the full report once. print it AND save it as txt (like run_all_eval does)
    parts = ["=" * 72,
             "LLM validator decisions (ground truth recomputed from each grid)",
             "goal % / val_reward need episode rollouts -> see run_all_eval results",
             "=" * 72,
             format_decision_table(summary)]
    rtable = format_reasoning_table(summary)
    if rtable:
        parts += ["", "=" * 72, "Reasoning quality (explain logs)", "=" * 72, rtable]
    if args.examples:
        ex = examples_report(df, args.examples)
        if ex:
            parts.append(ex)
    report = "\n".join(parts)
    print("\n" + report)

    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sum_path = os.path.join(out_dir, f"llm_explain_summary_{ts}.txt")
    rows_path = os.path.join(out_dir, f"llm_explain_rows_{ts}.csv")
    with open(sum_path, "w") as f:
        f.write(report + "\n")
    df.to_csv(rows_path, index=False)
    print(f"\nSaved summary to     {sum_path}")
    print(f"Annotated rows CSV:  {rows_path}")


if __name__ == "__main__":
    main()
