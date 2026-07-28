"""A lava-blind proposer that localizes by scan-matching and plans to the goal with BFS, remembering lava from validator rejections."""

from collections import deque
from typing import Any, Dict

import torch
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from env import ProposerAction, ValidatorAction, egocentric_view


# (row, col) offsets for UP, RIGHT, DOWN, LEFT
_DIR_OFFSET = {
    0: (-1, 0),  # UP
    1: (0, +1),  # RIGHT
    2: (+1, 0),  # DOWN
    3: (0, -1),  # LEFT
}

# exploratory forward chance when no lava-free path exists
# only relevant under an imperfect validator
_NO_PATH_FORWARD_PROB = 0.4


class PerfectProposerRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos: tuple[int, int] | None = None
        self._dir: int | None = None
        # cell -> step, for timestamping lava flags
        self._known_lava: dict[tuple[int, int], int] = {}
        self._last_action: int | None = None
        self._step: int = 0
        # RNG for the no-path exploratory step
        self._rng = np.random.default_rng()

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs_env = batch[SampleBatch.OBS]["env"]
        validator_action = batch[SampleBatch.OBS]["validator_action"]
        actions = [
            self._get_action(obs_env[i], validator_action[i])
            for i in range(len(obs_env))
        ]
        return {SampleBatch.ACTIONS: batch_func(actions)}
    
    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _pos_from_goal(er, ec, d, size):
        """Recover the agent's world position from where it sees the goal."""
        # agent sits at (R, R) in the ego view
        R = size - 1

        # goal's forward/right offset in the ego grid
        ego_fwd = R - er
        ego_rt = ec - R

        # world unit vectors for forward and right
        f = _DIR_OFFSET[d]
        rt = _DIR_OFFSET[(d+1)%4]

        # ego offsets -> world offsets goal minus agent
        d_row = ego_fwd * f[0] + ego_rt * rt[0]
        d_col = ego_fwd * f[1] + ego_rt * rt[1]

        # subtract offset from the fixed goal position
        return (size - d_row, size - d_col)

    @staticmethod
    def _border_walls(size):
        """Build the static outer wall border around the grid."""
        # interior plus a 1-tile wall border
        s = size + 2
        w = np.zeros((s, s), dtype=np.float32)

        # top and bottom walls
        w[0, :] = 1
        w[-1, :] = 1

        # left and right walls
        w[:, 0] = 1
        w[:, -1] = 1

        return w
        
    
    def _get_action(self, obs: torch.Tensor, validator_action: torch.Tensor) -> int:
        """Pick the next action from the egocentric view and validator feedback."""
        obs_np = obs.detach().cpu().numpy()
        size = int(obs_np.shape[0])

        # advance clock so lava flags age correctly
        self._step += 1

        disobeyed = (torch.argmax(validator_action).item() == ValidatorAction.disobey.value)

        # apply validator feedback before re-localizing
        if self._pos is not None and self._last_action is not None:
            self._apply_feedback(disobeyed, size)

        cands = self._candidates_from_obs(obs_np, size)

        # a single candidate pins the absolute state
        if len(cands) == 1:
            loc_pos, loc_dir = cands[0]

            # a big jump means a respawn; drop stale lava
            if self._pos is not None and self._manhattan(self._pos, loc_pos) > 1:
                self._known_lava = {}

            self._pos, self._dir = loc_pos, loc_dir

        elif self._pos is None:
            # not yet localized: rotate to gather more view
            self._last_action = ProposerAction.turn_right.value
            return self._last_action

        # BFS to goal avoiding known lava
        goal = (size, size)
        next_cell = self._bfs_next(self._pos, goal, size, set(self._known_lava))

        if next_cell is None and self._pos != goal:
            # no path: relax lava flags one by one
            next_cell = self._path_with_decayed_lava(self._pos, goal, size)

            if next_cell is None:
                # boxed in: sometimes push forward anyway
                action = (ProposerAction.forward.value
                          if self._rng.random() < _NO_PATH_FORWARD_PROB
                          else ProposerAction.turn_right.value)
                self._last_action = action
                return action

        # step toward next cell, or forward if already at goal
        action = (ProposerAction.forward.value if next_cell is None
                  else self._action_toward(next_cell))

        self._last_action = action
        return action

    def _path_with_decayed_lava(
        self, start: tuple[int, int], goal: tuple[int, int], size: int
    ) -> tuple[int, int] | None:
        """Drop oldest lava flags one at a time until BFS finds a path; return its next cell."""
        oldest_first = sorted(self._known_lava, key=self._known_lava.get)
        blocked = set(self._known_lava)
        for stale in oldest_first:
            blocked.discard(stale)
            nxt = self._bfs_next(start, goal, size, blocked)
            if nxt is not None:
                return nxt
        return None
        

    
    
    def _apply_feedback(self, disobeyed: bool, size: int)-> None:
        """Update the tracked position and lava map from the validator's verdict."""
        if self._last_action == ProposerAction.forward.value:
            target = self._forward_cell()
            in_grid = 1 <= target[0] <= size and 1 <= target[1] <= size

            if disobeyed and in_grid:
                # blocked forward: target is lava, stamp for decay
                self._known_lava[target] = self._step
            elif not disobeyed and in_grid:
                # allowed forward: target is safe, clear any flag
                self._known_lava.pop(target, None)
                self._pos = target

        elif not disobeyed:
            # turns always succeed; rotate the compass
            if self._last_action == ProposerAction.turn_left.value:
                self._dir = (self._dir - 1) % 4
            elif self._last_action == ProposerAction.turn_right.value:
                self._dir = (self._dir + 1) % 4
                
    def _candidates_from_obs(self, obs_np: np.ndarray, size: int):
        """Return the absolute states whose simulated view matches the observation."""
        walls = self._border_walls(size)
        goal = (size, size)
        R = size - 1

        # wall (channel 0) and goal (channel 2) planes
        obs_w = obs_np[..., 0]
        obs_g = obs_np[..., 2]

        goal_idx = np.argwhere(obs_g == 1.0)

        if len(goal_idx) > 0:
            # goal visible: one seed per direction
            er, ec = int(goal_idx[0][0]), int(goal_idx[0][1])
            seeds = []
            for d in range(4):
                pos = self._pos_from_goal(er, ec, d, size)
                # keep only in-bounds positions
                if 1 <= pos[0] <= size and 1 <= pos[1] <= size:
                    seeds.append((pos, d))
        else:
            # goal hidden: try every cell and orientation
            seeds = [((r, c), d)
                     for d in range(4)
                     for r in range(1, size + 1)
                     for c in range(1, size + 1)]

        output = []
        for pos, d in seeds:
            pred = egocentric_view(walls, pos, d, goal, [], R, include_lava=False)

            # keep seeds whose walls and goal match
            if np.array_equal(pred[..., 0], obs_w) and np.array_equal(pred[..., 2], obs_g):
                output.append((pos, d))

        return output
    
    
    # world cell in front of the agent
    def _forward_cell(self) -> tuple[int, int]:
        dr, dc = _DIR_OFFSET[self._dir]
        return (self._pos[0] + dr, self._pos[1] + dc)

    # action that moves toward next_cell
    def _action_toward(self, next_cell: tuple[int, int]) -> int:
        dr = next_cell[0] - self._pos[0]
        dc = next_cell[1] - self._pos[1]
        desired_dir = None
        for d, (ddr, ddc) in _DIR_OFFSET.items():
            if (ddr, ddc) == (dr, dc):
                desired_dir = d
                break
        if desired_dir is None:
            return ProposerAction.forward.value
        if desired_dir == self._dir:
            return ProposerAction.forward.value
        diff = (desired_dir - self._dir) % 4
        if diff == 1:
            return ProposerAction.turn_right.value
        if diff == 3:
            return ProposerAction.turn_left.value
        # 180 turn: default to right
        return ProposerAction.turn_right.value

    @staticmethod
    def _bfs_next(
        start: tuple[int, int],
        goal: tuple[int, int],
        size: int,
        blocked: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if start == goal:
            return None
        parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                node = cur
                while parent[node] != start:
                    node = parent[node]
                return node
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (1 <= nxt[0] <= size and 1 <= nxt[1] <= size):
                    continue
                if nxt in blocked or nxt in parent:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None
