import numpy as np
import torch

from env import egocentric_view, ProposerAction, ValidatorAction
from rl_modules.perfect_proposer import PerfectProposerRLM

_OBEY = torch.tensor([1.0, 0.0])      # one-hot, obey = 0
_DISOBEY = torch.tensor([0.0, 1.0])   # one-hot, disobey = 1


def _bare_proposer():
    # Build without RLModule.__init__ (no Ray spec needed for these unit tests).
    p = PerfectProposerRLM.__new__(PerfectProposerRLM)
    p._pos = None
    p._dir = None
    p._known_lava = {}          # cell -> step flagged
    p._last_action = None
    p._step = 0
    p._rng = np.random.default_rng(0)
    return p


def _view(size, pos, d):
    walls = PerfectProposerRLM._border_walls(size)
    return egocentric_view(walls, pos, d, (size, size), [], size - 1, include_lava=False)


def _all_poses(size):
    return [((r, c), d)
            for d in range(4)
            for r in range(1, size + 1)
            for c in range(1, size + 1)
            if (r, c) != (size, size)]


def _unique_goal_visible_poses(size):
    p = _bare_proposer()
    out = []
    for pose in _all_poses(size):
        v = _view(size, *pose)
        if v[..., 2].sum() > 0 and p._candidates_from_obs(v, size) == [pose]:
            out.append(pose)
    return out


class TestBFS:
    """Unit tests for the BFS planner inside the PerfectProposer.

    These check the static _bfs_next helper directly and don't
    require constructing an RLModule or a torch obs tensor.
    """

    def test_returns_none_when_at_goal(self):
        assert PerfectProposerRLM._bfs_next((2, 2), (2, 2), size=3, blocked=set()) is None

    def test_returns_next_step_not_full_path(self):
        # On an empty 3x3, from (1,1) to (3,3),
        # the first move must be to a neighbor of (1,1), either (1,2) or (2,1).
        nxt = PerfectProposerRLM._bfs_next((1, 1), (3, 3), size=3, blocked=set())
        assert nxt in {(1, 2), (2, 1)}

    def test_routes_around_lava(self):
        # Direct path (1,1) -> (1,2) -> (1,3) -> ... is blocked by lava at (1,2)
        # BFS should pick (2,1) instead
        nxt = PerfectProposerRLM._bfs_next(
            (1, 1), (3, 3), size=3, blocked={(1, 2)}
        )
        assert nxt == (2, 1)

    def test_returns_none_when_unreachable(self):
        # Wall of lava across row 2 cuts the agent off from the goal
        nxt = PerfectProposerRLM._bfs_next(
            (1, 1), (3, 3), size=3, blocked={(2, 1), (2, 2), (2, 3)}
        )
        assert nxt is None

    def test_respects_grid_bounds(self):
        # On a 1x1 grid the only cell is (1,1), which is also the goal
        assert PerfectProposerRLM._bfs_next((1, 1), (1, 1), size=1, blocked=set()) is None


class TestLocalization:
    def test_true_pose_always_recovered(self):
        for size in (3, 4, 5):
            p = _bare_proposer()
            for pose in _all_poses(size):
                cands = p._candidates_from_obs(_view(size, *pose), size)
                assert pose in cands

    def test_unique_iff_goal_visible(self):
        size = 4
        p = _bare_proposer()
        for pose in _all_poses(size):
            v = _view(size, *pose)
            cands = p._candidates_from_obs(v, size)
            if v[..., 2].sum() > 0:
                assert cands == [pose]
            else:
                assert len(cands) > 1

    def test_localizes_from_arbitrary_spawn(self):
        size = 5
        (pos, d) = _unique_goal_visible_poses(size)[0]
        p = _bare_proposer()
        p._get_action(torch.tensor(_view(size, pos, d)), _OBEY)
        assert (p._pos, p._dir) == (pos, d)

    def test_rotates_when_goal_not_visible_and_unlocalized(self):
        size = 4
        hidden = next(pose for pose in _all_poses(size)
                      if _view(size, *pose)[..., 2].sum() == 0)
        p = _bare_proposer()
        action = p._get_action(torch.tensor(_view(size, *hidden)), _OBEY)
        assert action == ProposerAction.turn_right.value
        assert p._pos is None


class TestTeleportAndLavaMemory:
    def test_teleport_clears_lava(self):
        size = 5
        uniq = _unique_goal_visible_poses(size)
        a = uniq[0]
        b = next(x for x in uniq if PerfectProposerRLM._manhattan(a[0], x[0]) > 1)

        p = _bare_proposer()
        p._get_action(torch.tensor(_view(size, *a)), _OBEY)
        assert p._pos == a[0]

        p._known_lava = {(2, 2): 0}
        p._last_action = None  # isolate from feedback
        p._get_action(torch.tensor(_view(size, *b)), _OBEY)
        assert p._pos == b[0]
        assert p._known_lava == {}

    def test_disobeyed_forward_marks_lava(self):
        p = _bare_proposer()
        p._pos, p._dir = (1, 1), 1  # facing RIGHT -> forward is (1,2)
        p._last_action = ProposerAction.forward.value
        p._apply_feedback(disobeyed=True, size=5)
        assert (1, 2) in p._known_lava
        assert p._pos == (1, 1)

    def test_obeyed_forward_clears_false_lava(self):
        p = _bare_proposer()
        p._pos, p._dir = (1, 1), 1
        p._last_action = ProposerAction.forward.value
        p._known_lava = {(1, 2): 0}
        p._apply_feedback(disobeyed=False, size=5)
        assert (1, 2) not in p._known_lava
        assert p._pos == (1, 2)


class TestLavaDecay:
    def test_decay_relaxes_oldest_flag_first(self):
        # On a 3x3, (1,1)'s only neighbors are (1,2) and (2,1). 
        # we flag both walls off the goal. The older flag should be decayed first to open that route.
        p = _bare_proposer()
        p._pos = (1, 1)
        p._known_lava = {(1, 2): 5, (2, 1): 1}  # (2,1) is older
        nxt = p._path_with_decayed_lava((1, 1), (3, 3), size=3)
        assert nxt == (2, 1)

    def test_decay_returns_none_when_nothing_to_relax(self):
        p = _bare_proposer()
        p._known_lava = {}
        assert p._path_with_decayed_lava((1, 1), (3, 3), size=3) is None
