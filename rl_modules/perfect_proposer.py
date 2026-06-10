"""BFS-planning proposer with scan-match localization and lava memory.

Due to its limited, lava-blind egocentric view, the agent tracks its absolute 
state `(pos, dir)` through the validator feedback and visual 
scan-matching against static walls and the goal. 

It uses BFS to plan paths to the goal, dynamically flagging blocked cells as 
lava whenever the validator rejects a forward move. Resets are detected by checking 
large coordinate jumps, and we clear the stale lava memory. If no safe path is found, it 
explores, or plans a BFS to the goal while relying on the validator as a safety net.
"""

from collections import deque
from typing import Any, Dict

import torch
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from env import ProposerAction, ValidatorAction, egocentric_view


# dir offsets in (row, col) format for UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3 respectively
_DIR_OFFSET = {
    0: (-1, 0),  # UP
    1: (0, +1),  # RIGHT
    2: (+1, 0),  # DOWN
    3: (0, -1),  # LEFT
}

# Probability of taking an exploratory forward step when BFS finds no lava-free path to the goal
# Only activated under an imperfect validator
_NO_PATH_FORWARD_PROB = 0.4


class PerfectProposerRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos: tuple[int, int] | None = None
        self._dir: int | None = None
        # cell to step dict so that we can timestamp the lava flags
        self._known_lava: dict[tuple[int, int], int] = {}
        self._last_action: int | None = None
        # a counter used to timestamp lava flags
        self._step: int = 0
        # RNG for the exploratory forward step in the no-path case
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
        """
        Calculates the agent's absolute world coordinates from the goal's position in the egocentric view.
        
        Since the goal is fixed at world coordinates `(size, size)`, the agent deduces its location by 
        transforming the goal's relative distance (forward and right) into absolute map vectors using 
        its current orientation `d`, and subtracting this offset from the goal's known world position.
        
        Args:
            er (int): Goal row in the agent's egocentric view.
            ec (int): Goal column in the agent's egocentric view.
            d (int): Agent's absolute orientation (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).
            size (int): Grid world size.
            
        Returns:
            tuple[int, int]: Absolute world coordinates (row, column) of the agent.
        """
        # The agent is always placed at the bottom center of the egocentric view, at (R, R)
        R = size - 1
        
        # Calculate how many cells ahead (forward) and to the right the goal is 
        # relative to the agent in the agent's egocentric grid.
        ego_fwd = R - er
        ego_rt = ec - R
        
        # Retrieve the world-coordinate unit vectors for the agent's forward and right directions
        f = _DIR_OFFSET[d]
        rt = _DIR_OFFSET[(d+1)%4]
        
        # Translate the egocentric distances into absolute world coordinate differences
        # d_row = (goal_world_row - agent_world_row)
        d_row = ego_fwd * f[0] + ego_rt * rt[0]
        # d_col = (goal_world_col - agent_world_col)
        d_col = ego_fwd * f[1] + ego_rt * rt[1]
        
        # The true world position of the goal is statically fixed at (size, size).
        # We subtract the calculated offsets from the goal to get the agent's true world position.
        return (size - d_row, size - d_col)

    @staticmethod
    def _border_walls(size):
        """
        Reconstructs the environment's static outer wall boundaries.
        
        Because the proposer only receives a limited egocentric view, it can use this 
        to build an internal representation of the full absolute map to predict views or 
        plan around absolute boundaries. 
        
        Args:
            size (int): The dimension of the playable interior grid.
            
        Returns:
            np.ndarray: A (size + 2) x (size + 2) float32 array where the outer edges 
                        are 1.0 (walls) and the interior is 0.0 (empty space).
        """
        # Overall map footprint includes the playable size plus 1 tile thick walls on each side
        s = size + 2
        w = np.zeros((s, s), dtype=np.float32)
        
        # Draw the top and bottom boundary walls
        w[0, :] = 1
        w[-1, :] = 1
        
        # Draw the left and right boundary walls
        w[:, 0] = 1
        w[:, -1] = 1
        
        return w
        
    
    def _get_action(self, obs: torch.Tensor, validator_action: torch.Tensor) -> int:
        """
        Determines the next action for the agent based on its egocentric observation and the 
        validator's feedback on the previous action.
        
        The agent uses a combination of validator feedback to learn about the lava positions and 
        visual observation matches to localize the absolute position of itself, before employing a BFS 
        planner to navigate towards the goal.
        
        Args:
            obs (torch.Tensor): The egocentric view tensor observation.
            validator_action (torch.Tensor): The action chosen by the validator (obey/disobey) 
                                             in response to the proposer's last action.
                                             
        Returns:
            int: The chosen discrete action index based on the ProposerAction enum.
        """
        obs_np = obs.detach().cpu().numpy()
        size = int(obs_np.shape[0])

        # Advance the decision clock so lava flags added this step are aged correctly
        self._step += 1

        # Evaluate if the validator blocked the last proposed action
        disobeyed = (torch.argmax(validator_action).item() == ValidatorAction.disobey.value)
        
        # If we have tracked a position and previously taken an action, 
        # apply the validator's feedback to update our internal position and lava map.
        if self._pos is not None and self._last_action is not None:
            self._apply_feedback(disobeyed, size)
            
        # triangulate the current absolute state (position and direction) 
        cands = self._candidates_from_obs(obs_np, size)
        
        # If there is exactly one valid candidate that perfectly explains our view
        # then we can pinpoint the absolute state
        if len(cands) == 1:
            loc_pos, loc_dir = cands[0]
            
            # If the calculated location implies an unexpected leap (> 1 Manhattan distance), 
            # we likely hit a fresh episode boundary and respawned. Then we clear the lava memory
            if self._pos is not None and self._manhattan(self._pos, loc_pos) > 1:
                self._known_lava = {}
                
            self._pos, self._dir = loc_pos, loc_dir
            
        elif self._pos is None:
            # If we haven't locked our position yet and couldn't pinpoint it from vision alone,
            # rotate in place to gather more visual information and re-evaluate next step.
            self._last_action = ProposerAction.turn_right.value
            return self._last_action

        # Use BFS to find the shortest path to the goal (size, size) that avoids
        # every cell we've recorded as lava through validator rejections.
        goal = (size, size)
        next_cell = self._bfs_next(self._pos, goal, size, set(self._known_lava))

        if next_cell is None and self._pos != goal:
            # if BFS to the goal is infeasible, we drop the lava flags one by one
            next_cell = self._path_with_decayed_lava(self._pos, goal, size)

            if next_cell is None:
                # if there is still no path, the agent is actually boxed in, 
                # then the agent invokes a probability to move forward
                action = (ProposerAction.forward.value
                          if self._rng.random() < _NO_PATH_FORWARD_PROB
                          else ProposerAction.turn_right.value)
                self._last_action = action
                return action

        # Move towards the next cell, or step forward if already at the goal.
        action = (ProposerAction.forward.value if next_cell is None
                  else self._action_toward(next_cell))

        self._last_action = action
        return action

    def _path_with_decayed_lava(
        self, start: tuple[int, int], goal: tuple[int, int], size: int
    ) -> tuple[int, int] | None:
        """Drop the oldest lava flags one at a time, retrying BFS until a path to the
        goal appears, and return that path's next cell"""
        oldest_first = sorted(self._known_lava, key=self._known_lava.get)
        blocked = set(self._known_lava)
        for stale in oldest_first:
            blocked.discard(stale)
            nxt = self._bfs_next(start, goal, size, blocked)
            if nxt is not None:
                return nxt
        return None
        

    
    
    def _apply_feedback(self, disobeyed: bool, size: int)-> None:
        """
        Updates the proposer's internal tracking of its absolute position and orientation based 
        on whether the validator permitted or blocked its last action.
        
        Because the proposer is "lava-blind", it uses the validator's disobedience to "discover" lava. 
        If a forward movement is blocked by the validator, the proposer assumes the target cell contains 
        lava and adds it to its internal map of known obstacles.
        
        Args:
            disobeyed (bool): True if the validator rejected the last proposed action, False if it was executed.
            size (int): The grid dimension, used to ensure the agent doesn't track movements out of bounds.
        """
        if self._last_action == ProposerAction.forward.value:
            # Calculate the absolute coordinate the agent attempted to enter
            target = self._forward_cell()
            in_grid = 1 <= target[0] <= size and 1 <= target[1] <= size
            
            if disobeyed and in_grid:
                # The validator blocked the forward movement
                # so the proposer assumes the target cell must be dangerous lava.
                # Stamp it with the current step so stale flags can be decayed
                self._known_lava[target] = self._step
            elif not disobeyed and in_grid:
                # The validator permitted the forward movement,
                # so the target cell is considered safe.
                # If we had previously flagged it as lava, clear it
                self._known_lava.pop(target, None)
                self._pos = target
                
        elif not disobeyed:
            # Turning actions are always executed successfully
            # Update the agent's internal compass direction using modular arithmetic (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).
            if self._last_action == ProposerAction.turn_left.value:
                self._dir = (self._dir - 1) % 4
            elif self._last_action == ProposerAction.turn_right.value:
                self._dir = (self._dir + 1) % 4
                
    def _candidates_from_obs(self, obs_np: np.ndarray, size: int):
        """
        Deduces a list of possible absolute states (position, direction) the agent might be in, 
        given its current egocentric observation.
        
        This works by generating "candidate" states and simulating what the agent would see from 
        those states. If the simulated view of the walls and goal matches the actual observation, 
        the state is considered valid.
        
        Args:
            obs_np (np.ndarray): The current egocentric observation array.
            size (int): The dimension of the inner grid.
            
        Returns:
            list[tuple[tuple[int, int], int]]: A list of valid candidates, where each candidate
                                               is a tuple containing the absolute position (row, col)
                                               and the direction `d`.
        """
        # Reconstruct the absolute world walls and goal to simulate views against
        walls = self._border_walls(size)
        goal = (size, size)
        R = size - 1
        
        # Extract the wall (channel 0) and goal (channel 2) matrices from the observation
        obs_w = obs_np[..., 0]
        obs_g = obs_np[..., 2]
        
        # Find where the goal is in the egocentric view, if it's visible
        goal_idx = np.argwhere(obs_g == 1.0)
        
        if len(goal_idx) > 0:
            # If the goal is visible, we can directly calculate the agent's position 
            # for each of the 4 possible current directions and reduce the candidate space
            er, ec = int(goal_idx[0][0]), int(goal_idx[0][1])
            seeds = []
            for d in range(4):
                pos = self._pos_from_goal(er, ec, d, size)
                # Only keep the calculated position if it falls within the grid bounds
                if 1 <= pos[0] <= size and 1 <= pos[1] <= size:
                    seeds.append((pos, d))
        else:
            # If the goal isn't visible, we must blindly consider every single 
            # possible cell and orientation in the entire grid as a potential candidate seed.
            seeds = [((r, c), d)
                     for d in range(4)
                     for r in range(1, size + 1)
                     for c in range(1, size + 1)]
                     
        output = []
        # Filter the generated seeds by simulating agent vision
        for pos, d in seeds:
            # Simulate what the egocentric view should look like from this candidate state
            pred = egocentric_view(walls, pos, d, goal, [], R, include_lava=False)
            
            # If the simulated walls and simulated goal exactly match the real observation,
            # then this candidate state is a highly plausible location for the agent.
            if np.array_equal(pred[..., 0], obs_w) and np.array_equal(pred[..., 2], obs_g): 
                output.append((pos, d))
                
        return output
    
    
    # Given the current tracked position and direction, return the cell in front of the agent in world coordinates
    def _forward_cell(self) -> tuple[int, int]:
        dr, dc = _DIR_OFFSET[self._dir]
        return (self._pos[0] + dr, self._pos[1] + dc)

    # Given the next cell to move to in world coordinates, determine the action needed to move toward it
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
        # 180 degree turn, can choose either way, choose right turn by default
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
