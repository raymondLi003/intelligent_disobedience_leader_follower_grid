from enum import IntEnum

import gymnasium as gym
import numpy as np
import pygame
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class EnvironmentAction(IntEnum):
    NO_OP = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    MOVE_FORWARD = 3


class ProposerAction(IntEnum):
    forward = 0
    turn_left = 1
    turn_right = 2


class ValidatorAction(IntEnum):
    obey = 0
    disobey = 1


def approve_nullify_operation_protocol(
        proposer_action: ProposerAction,
        validator_action: ValidatorAction,
) -> EnvironmentAction:
    if validator_action == ValidatorAction.obey:
        if proposer_action == ProposerAction.forward:
            return EnvironmentAction.MOVE_FORWARD
        elif proposer_action == ProposerAction.turn_left:
            return EnvironmentAction.TURN_LEFT
        elif proposer_action == ProposerAction.turn_right:
            return EnvironmentAction.TURN_RIGHT
        else:
            raise ValueError(f"Invalid proposer action: {proposer_action}")
    elif validator_action == ValidatorAction.disobey:
        return EnvironmentAction.NO_OP
    else:
        raise ValueError(f"Invalid validator action: {validator_action}")

def egocentric_view(
    walls: np.ndarray,
    agent_pos: tuple[int, int],
    agent_dir: int,
    goal_pos: tuple[int, int],
    lava_positions,
    view_radius:int,
    include_lava: bool,
)-> np.ndarray:
    """Cropped egocentric view shared by the env and perfect proposer so predicted views match real ones."""
    s = walls.shape[0]
    full = np.zeros((s, s, 4), dtype = np.float32)
    full[:, :, 0] = walls
    full[agent_pos[0], agent_pos[1], 1] = 1.0
    full[goal_pos[0], goal_pos[1], 2] = 1.0
    for r, c in lava_positions:
        full[r, c, 3] = 1.0

    pad = view_radius
    wall_padded = np.pad(full[:, :, 0], ((pad,pad), (pad, pad)), mode = "constant", constant_values= 1.)
    others_padded = np.pad(full[:,:,1:], ((pad,pad), (pad,pad), (0,0)), mode= "constant", constant_values= 0.)
    padded = np.concatenate((wall_padded[..., None], others_padded), axis=-1)
    
    # center on agent
    row = agent_pos[0] + pad
    col = agent_pos[1] + pad
    local = padded[row - pad: row + pad + 1, col - pad: col + pad + 1, :]

    # rotate so agent faces up
    local = np.rot90(local, k = agent_dir, axes= (0,1))

    # keep agent's row and everything ahead
    center = local.shape[1]//2
    local = local[0:pad + 1, center - pad: center + pad + 1, :]
    
    if not include_lava:
        local = np.delete(local, 3, axis=-1)
    
    return local.astype(np.float32).copy()
        
    

class GridWorldEnv(MultiAgentEnv):
    # Orientation encoding
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(
            self,
            size: int = 4,
            num_lava_tiles=2,
            max_steps: int | None = None,
            render: bool = False,
            record_render: bool = False,
            single_agent: bool = False,
            proposer_sees_lava: bool = False,
            randomize_spawn: bool = False,
            seed=None,
    ):
        super().__init__()
        assert num_lava_tiles >= 0

        self.size = size
        self._size_with_walls = size + 2
        self.num_lava_tiles = num_lava_tiles
        self.agent_view_radius = size - 1
        self.max_steps = max_steps
        self.proposer_sees_lava = proposer_sees_lava
        self.randomize_spawn = randomize_spawn
        self._steps = 0
        self.render = render

        self.rng = np.random.default_rng(seed)
        self._build_static_grid()

        self.single_agent = single_agent
        if single_agent:
            self.agents = self.possible_agents = ["single_agent"]
        else:
            self.agents = self.possible_agents = ["proposer", "validator"]
        self.agent_pos = None
        self.agent_dir = None
        self.done = False
        self._operation_protocol = approve_nullify_operation_protocol

        if single_agent:
            self.action_spaces = {
                "single_agent": gym.spaces.Discrete(len(EnvironmentAction)),
            }
            self.observation_spaces = {
                "single_agent": gym.spaces.Box(low=0, high=1,
                                               shape=(self.agent_view_radius + 1, self.agent_view_radius * 2 + 1, 4),
                                               dtype=np.float32),
            }
        else:
            self.action_spaces = {
                "proposer": gym.spaces.Discrete(len(ProposerAction)),
                "validator": gym.spaces.Discrete(len(ValidatorAction)),
            }
            self.observation_spaces = {
                "proposer": gym.spaces.Dict({
                    "env": gym.spaces.Box(low=0, high=1,
                                          shape=(self.agent_view_radius + 1, self.agent_view_radius * 2 + 1,
                                                 3 if not proposer_sees_lava else 4),
                                          dtype=np.float32),
                    "validator_action": gym.spaces.Box(low=0, high=1, shape=(len(ValidatorAction),), dtype=np.float32),
                }),
                "validator": gym.spaces.Dict({
                    "env": gym.spaces.Box(low=0, high=1,
                                          shape=(self.agent_view_radius + 1, self.agent_view_radius * 2 + 1, 4),
                                          dtype=np.float32),
                    "proposer_action": gym.spaces.Box(low=0, high=1, shape=(len(ProposerAction),), dtype=np.float32),
                }),
            }

        self._pygame_initialized = False
        self._record_render = record_render
        self._frames = []

    def _build_static_grid(self):
        self.walls = np.zeros((self._size_with_walls, self._size_with_walls), dtype=np.uint8)

        self.walls[0, :] = 1
        self.walls[-1, :] = 1
        self.walls[:, 0] = 1
        self.walls[:, -1] = 1

        self.goal_pos = np.array([self._size_with_walls - 2, self._size_with_walls - 2], dtype=np.int32)
        if self.num_lava_tiles > 0:
            self.lava_positions = self._generate_lava_positions()
        else:
            self.lava_positions = []

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        self.done = False
        self._steps = 0

        self.agent_pos = np.array([1, 1], dtype=np.int32)
        self.agent_dir = self.RIGHT

        if options is not None:
            if "lava_positions" in options:
                assert isinstance(options["lava_positions"], list) or isinstance(options["lava_positions"], tuple)
                assert len(options["lava_positions"]) == self.num_lava_tiles
                self.lava_positions = []
                for lava_position in options["lava_positions"]:
                    assert isinstance(lava_position, tuple)
                    assert len(lava_position) == 2
                    self.lava_positions.append((lava_position[0] + 1, lava_position[1] + 1))
        elif self.num_lava_tiles > 0:
            self.lava_positions = self._generate_lava_positions()

        # randomized spawn for training, off in eval
        if self.randomize_spawn:
            lava_set = set(tuple(p) for p in self.lava_positions)
            goal_cell = tuple(self.goal_pos.tolist())
            candidates = [
                (r, c)
                for r in range(1, self._size_with_walls - 1)
                for c in range(1, self._size_with_walls - 1)
                if (r, c) not in lava_set and (r, c) != goal_cell
            ]
            idx = int(self.rng.integers(0, len(candidates)))
            self.agent_pos = np.array(candidates[idx], dtype=np.int32)
            self.agent_dir = int(self.rng.integers(0, 4))

        if self.render:
            self.render_env()

        if self.single_agent:
            return {"single_agent": self._get_observation("single_agent")}, {}
        proposer_observation = self._get_observation("proposer")
        validator_one_hot = np.zeros(len(ValidatorAction), dtype=np.float32)
        validator_one_hot[ValidatorAction.obey] = 1.
        return {
            "proposer": {
                "env": proposer_observation,
                "validator_action": validator_one_hot,
            }
        }, {}

    def _generate_lava_positions(self) -> list[tuple[int, int]]:
        potential_lava_positions = [
            (row, column)
            for row in range(1, self._size_with_walls - 1)
            for column in range(1, self._size_with_walls - 1)
            if not (row == 1 and column == 1) and not (row == self.goal_pos[0] and column == self.goal_pos[1])
        ]

        self.rng.shuffle(potential_lava_positions)

        assert len(potential_lava_positions) >= self.num_lava_tiles + 1

        lava_position_selection = potential_lava_positions[:self.num_lava_tiles]
        while True:
            # Do not block start or goal
            if (1, 2) in lava_position_selection and (2, 1) in lava_position_selection:
                del potential_lava_positions[0]
                lava_position_selection = potential_lava_positions[:self.num_lava_tiles]
            elif ((self.goal_pos[0] - 1, self.goal_pos[1]) in lava_position_selection
                  and (self.goal_pos[0], self.goal_pos[1] - 1) in lava_position_selection):
                del potential_lava_positions[0]
                lava_position_selection = potential_lava_positions[:self.num_lava_tiles]
            else:
                break
        return lava_position_selection

    def _step_proposer(self, proposer_action: ProposerAction) -> MultiAgentDict:
        self._proposer_action = proposer_action
        return {}

    def _step_validator(self, validator_action: ValidatorAction) -> MultiAgentDict:
        self._validator_action = validator_action

        action = self._operation_protocol(self._proposer_action, self._validator_action)
        proposer_reward = self._step_single_agent(action)
        validator_reward = 0

        if action == EnvironmentAction.NO_OP:
            if self._proposer_action != ProposerAction.forward:
                validator_reward = -1
            else:
                forward_pos = self._forward_position()
                if tuple(forward_pos) in self.lava_positions:
                    validator_reward = 1
                else:
                    validator_reward = -1
        else:
            if self._proposer_action == ProposerAction.forward:
                if tuple(self.agent_pos) in self.lava_positions:
                    validator_reward = -1

        return {
            "proposer": proposer_reward,
            "validator": validator_reward,
        }

    def _step_single_agent(self, action: EnvironmentAction) -> float:
        if action == EnvironmentAction.TURN_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4

        elif action == EnvironmentAction.TURN_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4

        if action == EnvironmentAction.MOVE_FORWARD:
            forward_pos = self._forward_position()

            if self.walls[forward_pos[0], forward_pos[1]] == 0:
                self.agent_pos = forward_pos
                if tuple(self.agent_pos) in self.lava_positions:
                    self.done = True
                    return -1
                elif np.array_equal(self.agent_pos, self.goal_pos):
                    self.done = True
                    return 1
        elif action == EnvironmentAction.NO_OP:
            pass
        elif action not in (EnvironmentAction.TURN_LEFT, EnvironmentAction.TURN_RIGHT):
            raise ValueError("Invalid action.")

        return 0

    def step(
            self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        if "proposer" in action_dict:
            rewards = self._step_proposer(action_dict["proposer"])
            obs = self._get_observation("validator")
            proposer_action_one_hot = np.zeros(len(ProposerAction), dtype=np.float32)
            proposer_action_one_hot[self._proposer_action] = 1.

            obs = {
                "validator": {
                    "env": obs,
                    "proposer_action": proposer_action_one_hot,
                },
            }
        elif "validator" in action_dict:
            rewards = self._step_validator(action_dict["validator"])
            obs = self._get_observation("proposer")
            validator_action_one_hot = np.zeros(len(ValidatorAction), dtype=np.float32)
            validator_action_one_hot[self._validator_action] = 1.

            obs = {
                "proposer": {
                    "env": obs,
                    "validator_action": validator_action_one_hot,
                }
            }
        elif "single_agent" in action_dict:
            reward = self._step_single_agent(action_dict["single_agent"])
            rewards = {"single_agent": reward}
            obs = {
                "single_agent": self._get_observation("single_agent"),
            }
        else:
            raise ValueError("Invalid action.")
        self._steps += 1

        if self.render:
            self.render_env()

        terminated = {
            "__all__": self.done,
        }
        for agent_id in self.agents:
            terminated[agent_id] = self.done

        if self.max_steps is not None and self._steps >= self.max_steps:
            truncated = {"__all__": True}
            for agent_id in self.agents:
                truncated[agent_id] = True

            if not self.single_agent:
                if "proposer" not in obs:
                    obs["proposer"] = {
                        "env": self._get_observation("proposer"),
                        "validator_action": np.zeros(len(ValidatorAction), dtype=np.float32),
                    }
                if "validator" not in obs:
                    obs["validator"] = {
                        "env": self._get_observation("validator"),
                        "proposer_action": np.zeros(len(ProposerAction), dtype=np.float32),
                    }
        else:
            truncated = {"__all__": False}
            for agent_id in self.agents:
                truncated[agent_id] = False

        return (
            obs,
            rewards,
            terminated,
            truncated,
            {}
        )

    def _forward_position(self):
        row, column = self.agent_pos

        if self.agent_dir == self.UP:
            return np.array([row - 1, column])
        elif self.agent_dir == self.RIGHT:
            return np.array([row, column + 1])
        elif self.agent_dir == self.DOWN:
            return np.array([row + 1, column])
        elif self.agent_dir == self.LEFT:
            return np.array([row, column - 1])
        else:
            raise ValueError("Invalid agent direction.")

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """only the lava-blind proposer drops the lava channel."""
        include_lava = not (agent_id == "proposer" and not self.proposer_sees_lava)
        return egocentric_view(
            self.walls,
            (int(self.agent_pos[0]), int(self.agent_pos[1])),
            int(self.agent_dir),
            (int(self.goal_pos[0]), int(self.goal_pos[1])),
            self.lava_positions,
            self.agent_view_radius,
            include_lava,
        )
        

    def render_env(self) -> None:
        if not self._pygame_initialized:
            pygame.init()

            self._tile_size = 48
            self._mini_tile = 18
            self._margin = 2

            grid_w = self._size_with_walls * self._tile_size
            grid_h = self._size_with_walls * self._tile_size

            mini_size_w = (self.agent_view_radius * 2 + 1) * self._mini_tile
            mini_size_h = (self.agent_view_radius + 1) * self._mini_tile

            self._width = grid_w + mini_size_w + 40
            self._height = max(grid_h, mini_size_h + 60)

            self._screen = pygame.display.set_mode((self._width, self._height))
            pygame.display.set_caption("GridWorld")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("arial", 18)

            self._pygame_initialized = True

            # Colors
            self._color_wall = (0, 0, 0)
            self._color_floor = (200, 200, 200)
            self._color_floor_alt = (185, 185, 185)
            self._color_lava = (255, 120, 0)
            self._color_goal = (0, 180, 0)
            self._color_agent = (30, 60, 200)
            self._color_panel = (240, 240, 240)
            self._color_border = (120, 120, 120)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return

        self._screen.fill((255, 255, 255))

        ts = self._tile_size
        margin = self._margin

        # ============================================================
        # MAIN GLOBAL OBS
        # ============================================================
        for r in range(self._size_with_walls):
            for c in range(self._size_with_walls):
                x = c * ts
                y = r * ts
                rect = pygame.Rect(x, y, ts, ts)

                # Walls
                if self.walls[r, c] == 1:
                    pygame.draw.rect(self._screen, self._color_wall, rect)
                    continue

                # checker floor
                color = self._color_floor if (r + c) % 2 == 0 else self._color_floor_alt
                pygame.draw.rect(self._screen, color, rect)

                pygame.draw.rect(
                    self._screen,
                    (150, 150, 150),
                    rect,
                    width=margin,
                )

        # Lava
        for (r, c) in self.lava_positions:
            rect = pygame.Rect(c * ts, r * ts, ts, ts)
            pygame.draw.rect(self._screen, self._color_lava, rect.inflate(-8, -8), border_radius=6)

        # Goal
        if self.goal_pos is not None:
            r, c = self.goal_pos
            rect = pygame.Rect(c * ts, r * ts, ts, ts)
            pygame.draw.rect(self._screen, self._color_goal, rect.inflate(-10, -10), border_radius=8)

        # Agent triangle
        if self.agent_pos is not None:
            r, c = self.agent_pos
            cx = c * ts + ts / 2
            cy = r * ts + ts / 2
            size = ts * 0.35

            if self.agent_dir == self.UP:
                pts = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            elif self.agent_dir == self.RIGHT:
                pts = [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)]
            elif self.agent_dir == self.DOWN:
                pts = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
            elif self.agent_dir == self.LEFT:
                pts = [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)]
            else:
                pts = []

            if pts:
                pygame.draw.polygon(self._screen, self._color_agent, pts)

        # ============================================================
        # EGOCENTRIC OBS
        # ============================================================
        mini_obs = self._get_observation("validator")

        mini_tiles_h, mini_tiles_w, _ = mini_obs.shape
        mini_ts = self._mini_tile

        offset_x = self._size_with_walls * ts + 20
        offset_y = 40

        panel_w = mini_tiles_w * mini_ts
        panel_h = mini_tiles_h * mini_ts

        # Panel background
        panel_rect = pygame.Rect(offset_x - 10, offset_y - 30, panel_w + 20, panel_h + 40)
        pygame.draw.rect(self._screen, self._color_panel, panel_rect, border_radius=8)
        pygame.draw.rect(self._screen, self._color_border, panel_rect, width=2, border_radius=8)

        # Title
        title = self._font.render("Egocentric View", True, (0, 0, 0))
        self._screen.blit(title, (offset_x, offset_y - 24))

        # Draw minimap tiles
        for r in range(mini_tiles_h):
            for c in range(mini_tiles_w):
                tile = mini_obs[r, c]

                x = offset_x + c * mini_ts
                y = offset_y + r * mini_ts
                rect = pygame.Rect(x, y, mini_ts, mini_ts)

                # walls channel
                if tile[0] > 0.5:
                    pygame.draw.rect(self._screen, self._color_wall, rect)
                    continue

                color = (
                    self._color_floor
                    if (r + c) % 2 == 0
                    else self._color_floor_alt
                )
                pygame.draw.rect(self._screen, color, rect)

                pygame.draw.rect(
                    self._screen,
                    (150, 150, 150),
                    rect,
                    width=max(1, self._margin - 1),
                )

                # lava
                if tile[3] > 0.5:
                    pygame.draw.rect(
                        self._screen,
                        self._color_lava,
                        rect.inflate(-mini_ts * 0.35, -mini_ts * 0.35),
                        border_radius=3,
                    )

                # goal
                if tile[2] > 0.5:
                    pygame.draw.rect(
                        self._screen,
                        self._color_goal,
                        rect.inflate(-mini_ts * 0.35, -mini_ts * 0.35),
                        border_radius=3,
                    )

        # agent sits bottom-center facing up
        center_c = mini_tiles_w // 2
        agent_r = mini_tiles_h - 1

        cx = offset_x + center_c * mini_ts + mini_ts / 2
        cy = offset_y + agent_r * mini_ts + mini_ts / 2
        size = mini_ts * 0.4

        pts = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size),
        ]
        pygame.draw.polygon(self._screen, self._color_agent, pts)

        pygame.display.flip()
        self._clock.tick(30)

        if self._record_render:
            frame = pygame.surfarray.array3d(self._screen)
            frame = np.transpose(frame, (1, 0, 2))
            self._frames.append(frame)

    def __del__(self):
        pygame.quit()

    def save_video(self, path):
        import imageio
        if self._frames:
            with imageio.get_writer(path, fps=30) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
