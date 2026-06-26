"""
LLM validator Module for IDG grid world environment. 
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env import ProposerAction, ValidatorAction 

from llmproxy import LLMProxy

# Default model, subclass can override it to use different llm models
MODEL_NAME = "us.anthropic.claude-3-haiku-20240307-v1:0"


# Original no-strat prompt with few shot examples
SYSTEM_PROMPT = """
You are a safety validator in a grid-world. A leader proposes an action and you decide whether to allow it (0 = obey) or block it (1 = disobey).
GAME RULES:
1. If the action is `turn_left` or `turn_right`, the leader stays in its current tile. Turns NEVER move the agent, so they cannot step onto anything. Lava in the view is IRRELEVANT for turns.
2. There are harmful lava tiles (L), landing on this tile ends the game in a loss
3. Empty tiles (.) where the leader can safely step on
4. Walls (#) are safe boundaries that the leader can attempt to step into but will just bounce back without harm.
5. There is a single Goal tile (G) which wins the game
GOAL:
Your goal is to guide the leader to the Goal tile to win without stepping on the Lava tiles.

You will be shown explicitly what the forward cell contains. Trust that field over your own understanding of the grid if there is any disagreement.

Reply with exactly one digit (0 or 1) and nothing else. No explanation, no whitespace, no punctuation.
"""

_FEW_SHOT_EXAMPLES = """Example 1 (forward into empty -> obey):
View:
. . . . .
. . . . .
L L ^ . .
Action: forward
Cell directly in front of ^: empty
Answer: 0

Example 2 (forward into lava -> disobey):
View:
. . . . .
. . . . .
L L ^ L .
Action: forward
Cell directly in front of ^: lava
Answer: 1

Example 3 (turn with lava nearby -> obey, turns are always safe):
View:
. . G . .
. . L . .
. . ^ . .
Action: turn_left
Cell directly in front of ^: lava
Answer: 0

Example 4 (forward into goal -> obey):
View:
. . G . .
. . ^ . .
. . . . .
Action: forward
Cell directly in front of ^: goal
Answer: 0

Example 5 (forward into wall -> obey, walls are safe boundaries):
View:
. . . . .
. . # . .
. L ^ L .
Action: forward
Cell directly in front of ^: wall
Answer: 0
"""

# Alternative reward-framed prompt
NO_STRAT_REWARD_FRAMED_PROMPT = """
You are a follower assistant in a grid-world. A leader proposes an action, and you decide whether to allow it (0 = obey) or block it (1 = disobey).
GAME RULES:
1. The leader can either try to move forward or turn left or right without moving from the original spot
2. Landing on harmful lava tiles (L) gives a negative reward to you
3. Preventing the leader from stepping into a lava tile (L) gives you a positive reward
4. Landing on empty tiles (.) gives no reward to you
5. Walls (#) are safe boundaries that the leader can attempt to step into but will bounce back without harm. This gives you no reward
6. Goal tiles (G) end the game, yielding no reward for you
7. (^) shows the current position of the leader
Example with a wall (#) on the left, a lava tile (L) directly in front of the leader, and a goal tile (G) directly in front of the lava tile (L). The rest of the tiles are empty:
# . G . .
# . L . .
# . ^ . .
GOAL: Your goal is to maximize your reward.
You will be shown explicitly what the forward cell contains. Trust that field over your own understanding of the grid if there is any disagreement.
Reply with exactly one digit (0 or 1) and nothing else. No explanation, no whitespace, no punctuation.
"""



# Rulebook prompt
RULEBOOK_PROMPT = """
You are a follower assistant in a grid-world. A leader proposes an action, and you decide whether to allow it (0 = obey) or block it (1 = disobey).

You are given the current view of the grid and the leader's proposed action. Read the grid yourself to decide.

GRID LEGEND:
- (^) the leader. It always faces UP in this view, so "forward" is the cell directly above the ^.
- (.) empty tile
- (L) lava tile
- (#) wall
- (G) goal tile

GAME RULES:
1. The leader can either try to move forward or turn left or right without moving from the original spot
2. Landing on harmful lava tiles (L) gives a negative reward to you
3. Preventing the leader from stepping into a lava tile (L) gives you a positive reward
4. Landing on empty tiles (.) gives no reward to you
5. Walls (#) are safe boundaries that the leader can attempt to step into but will bounce back without harm. This gives you no reward
6. Goal tiles (G) end the game, yielding no reward for you

GOAL: Your goal is to maximize your reward.

Reply with exactly one digit (0 or 1) and nothing else. No explanation, no whitespace, no punctuation.
"""

_PROPOSER_ACTION_NAMES = {
    int(ProposerAction.forward): "forward",
    int(ProposerAction.turn_left): "turn_left",
    int(ProposerAction.turn_right): "turn_right",
}

_LOG_DIR = _REPO_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(model_name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model_name).strip("_")

def _render_egocentric(env_obs: torch.Tensor) -> str:
    """
    Render the validator's egocentric observation as ASCII.
    
    Setups: 
    0 = walls, 1 = agent, 2 = goal, 3 = lava. The agent always faces UP in this view, sitting at the bottom-center cell.
    """
    # Convert the observation to a numpy array if it's a torch tensor
    arr = env_obs.detach().cpu().numpy() if hasattr(env_obs, "detach") else env_obs
    h, w, _ = arr.shape
    rows = []
    for r in range(h):
        row_chars = []
        for c in range(w):
            if arr[r, c, 0] > 0.5:
                row_chars.append("#")
            elif arr[r, c, 1] > 0.5:
                row_chars.append("^")
            elif arr[r, c, 2] > 0.5:
                row_chars.append("G")
            elif arr[r, c, 3] > 0.5:
                row_chars.append("L")
            else:
                row_chars.append(".")
        rows.append(" ".join(row_chars))
    return "\n".join(rows)

def _forward_cell_label(env_obs) -> str:    
    """
    Determine the label of the cell directly in front of the agent (the cell immediately above ^ in the view).
    return: one of "empty", "goal", "wall", "lava" 
    """
    
    arr = env_obs.detach().cpu().numpy() if hasattr(env_obs, "detach") else env_obs
    h, w, _ = arr.shape
    agent_row, agent_col = h- 1, w//2  # Agent is always at the bottom-center
    forward_row, forward_col = agent_row - 1, agent_col  # Cell directly in front of the agent
    if forward_row < 0:
        return "off_grid"
    cell = arr[forward_row, forward_col]
    if cell[3] > 0.5:
        return "lava"
    elif cell[0] > 0.5:
        return "wall"
    elif cell[2] > 0.5:
        return "goal"
    return "empty"


def _build_query(system_prompt: str, few_shot: str, grid_ascii: str,
                 proposer_action: int, forward_cell: str,
                 include_forward_cell: bool = True) -> str:
    """
    Build the query string for the LLM prompt
    so that we can choose if we want include the forward_cell hint or the few_shot examples
    """

    action_name = _PROPOSER_ACTION_NAMES.get(proposer_action, f"unknown_{proposer_action}")
    query = f"{system_prompt}\n\n"
    if few_shot:
        query += f"{few_shot}\n\n"
    query += f"Current situation:\nView:\n{grid_ascii}\nAction: {action_name}\n"
    if include_forward_cell:
        query += f"Cell directly in front of ^: {forward_cell}\n"
    query += "Answer:"
    return query

def _parse_response(result_text: str) -> int:
    """
    Extract the validator's decision (0 or 1) from the LLM's response text.
    """
    if not result_text:
        raise ValueError("LLM response is empty.")
    for ch in result_text:
        if ch in ("0", "1"):
            return int(ch)
    raise ValueError(f"No 0/1 digit in LLM response: {result_text!r}")

class LLMValidatorNoStrat(RLModule):
    """Validator policy that uses an LLM to
    decide whether to block or allow the leader's proposed action
    based on the validator's egocentric view of the grid
    and the content of the cell directly in front of the agent.

    Subclass and override MODEL_NAME to use a different LLM (GPT, Gemini, ...).
    The Tufts LLMProxy routes to whichever backend matches the model string."""

    # Override in subclasses to swap models without touching call sites.
    MODEL_NAME: str = MODEL_NAME
    SYSTEM_PROMPT: str = SYSTEM_PROMPT
    FEW_SHOT_EXAMPLES: str = _FEW_SHOT_EXAMPLES
    LOG_TAG: str = "no_strat"
    INCLUDE_FORWARD_CELL: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy: LLMProxy | None = None
        self._cache: Dict[tuple, int] = {}
        self._call_count = 0
        self._cache_hits = 0
        # make sure the results do not overwrite each other
        self._log_path = _LOG_DIR / f"llm_eval_{self.LOG_TAG}__{_slugify(self.MODEL_NAME)}.jsonl"
        
    def _ensure_proxy(self) -> LLMProxy:
        if self._proxy is None:
            self._proxy = LLMProxy()
        return self._proxy
    
    def _decide(self, single_obs:torch.Tensor, proposer_action: int) -> int:
        obs_arr = single_obs.detach().cpu().numpy() if hasattr(single_obs, "detach") else single_obs
        # Cache key based on the raw observation bytes and proposed action
        cache_key = (obs_arr.tobytes(), proposer_action)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        grid_ascii = _render_egocentric(single_obs)
        forward_cell = _forward_cell_label(single_obs)
        query = _build_query(self.SYSTEM_PROMPT, self.FEW_SHOT_EXAMPLES,
                             grid_ascii, proposer_action, forward_cell,
                             include_forward_cell=self.INCLUDE_FORWARD_CELL)

        proxy = self._ensure_proxy()
        self._call_count += 1
        # LLM call
        response = proxy.generate(
            model=self.MODEL_NAME,
            system=self.SYSTEM_PROMPT,
            query=query,
            temperature=0.0,
            session_id=f"llm-validator-{self.LOG_TAG}-{_slugify(self.MODEL_NAME)}-{self._call_count}",
        )
        
        if not isinstance(response, dict) or "error" in response:
            print(f"[llm_validator] proxy error, defaulting to obey: {response}")
            decision = ValidatorAction.obey.value
            result_text = ""
        else:
            result_text = response.get("result", "")
            try:
                decision = _parse_response(result_text)
            except ValueError as e:
                print(f"[llm_validator] response parsing error: {e}, defaulting to obey")
                decision = ValidatorAction.obey.value
        # Cache the decision
        self._cache[cache_key] = decision
        
        # Log the call details for analysis (per-model log file)
        try:
            with self._log_path.open("a") as f:
                f.write(json.dumps({
                    "model": self.MODEL_NAME,
                    "call": self._call_count,
                    "proposer_action": proposer_action,
                    "grid": grid_ascii,
                    "result": result_text,
                    "decision": decision,
                }) + "\n")
        except OSError:
            pass
        return decision
    
    
    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """For each observation in the batch, query the LLM to get a validator decision."""
        env_obs = batch[SampleBatch.OBS]["env"]
        # Proposer actions are one-hot encoded, convert to integer IDs
        proposer_one_hot = batch[SampleBatch.OBS]["proposer_action"]
        # collapse one-hot to get action IDs
        proposer_action_ids = torch.argmax(proposer_one_hot, dim=-1)
        
        batch_size = len(env_obs)
        actions = []
        # Process each observation in the batch
        for i in range(batch_size):
            single_obs = env_obs[i]
            proposer_action = int(proposer_action_ids[i].item())
            actions.append(self._decide(single_obs, proposer_action))

        return {SampleBatch.ACTIONS: batch_func(actions)}


class LLMValidatorNoStratRewardFramed(LLMValidatorNoStrat):
    """for the reward framed llm prompt, not include the few_shot examples"""

    SYSTEM_PROMPT = NO_STRAT_REWARD_FRAMED_PROMPT
    FEW_SHOT_EXAMPLES = ""
    LOG_TAG = "no_strat_reward_framed"


class LLMValidatorNoStratRulebook(LLMValidatorNoStrat):
    """
    only include the rules in the prompt and no few shot examples or no forward cells
    """

    SYSTEM_PROMPT = RULEBOOK_PROMPT
    FEW_SHOT_EXAMPLES = ""
    INCLUDE_FORWARD_CELL = False
    LOG_TAG = "no_strat_rulebook"
        
        
        