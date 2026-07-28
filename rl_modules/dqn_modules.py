import torch
import tree
from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_PREDS, QF_NEXT_PREDS, QF_TARGET_NEXT_PREDS, ATOMS, \
    QF_LOGITS, QF_PROBS, QF_TARGET_NEXT_PROBS
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorStructType, TensorType


class DQNWithDictOBS(DefaultDQNTorchRLModule):
    @override(RLModule)
    def _forward_train(
            self, batch: dict[str, TensorType]
    ) -> dict[str, TensorStructType]:
        if self.inference_only:
            raise RuntimeError(
                "Trying to train a module that is not a learner module. Set the "
                "flag `inference_only=False` when building the module."
            )
        output = {}

        if self.uses_double_q:
            # single forward pass over current and next obs
            if isinstance(batch[Columns.OBS], dict):
                batch_base = {
                    Columns.OBS: tree.map_structure(
                        lambda obs, next_obs: torch.concat(
                            [obs, next_obs], dim=0
                        ),
                        batch[Columns.OBS],
                        batch[Columns.NEXT_OBS],
                    )
                }
            else:
                batch_base = {
                    Columns.OBS: torch.concat(
                        [batch[Columns.OBS], batch[Columns.NEXT_OBS]], dim=0
                    ),
                }
            # stateful: add input states
            if Columns.STATE_IN in batch:
                batch_base.update(
                    {
                        Columns.STATE_IN: tree.map_structure(
                            lambda t1, t2: torch.cat([t1, t2], dim=0),
                            batch[Columns.STATE_IN],
                            batch[Columns.NEXT_STATE_IN],
                        )
                    }
                )
        else:
            batch_base = {Columns.OBS: batch[Columns.OBS]}
            # stateful: add input state
            if Columns.STATE_IN in batch:
                batch_base.update({Columns.STATE_IN: batch[Columns.STATE_IN]})

        batch_target = {Columns.OBS: batch[Columns.NEXT_OBS]}

        # stateful: add target-pass states
        if Columns.NEXT_STATE_IN in batch:
            batch_target.update({Columns.STATE_IN: batch[Columns.NEXT_STATE_IN]})

        # Q-network forward passes.
        qf_outs = self.compute_q_values(batch_base)
        if self.uses_double_q:
            output[QF_PREDS], output[QF_NEXT_PREDS] = torch.chunk(
                qf_outs[QF_PREDS], chunks=2, dim=0
            )
        else:
            output[QF_PREDS] = qf_outs[QF_PREDS]
        qf_target_next_outs = self.forward_target(batch_target)
        output[QF_TARGET_NEXT_PREDS] = qf_target_next_outs[QF_PREDS]
        # distributional Q-learning
        if self.num_atoms > 1:
            # distribution support
            output[ATOMS] = qf_target_next_outs[ATOMS]
            output[QF_LOGITS] = qf_outs[QF_LOGITS]
            # current-state probabilities
            output[QF_PROBS] = qf_outs[QF_PROBS]
            # next-state target probabilities
            output[QF_TARGET_NEXT_PROBS] = qf_target_next_outs[QF_PROBS]

        # stateful: add output states
        if Columns.STATE_OUT in qf_outs:
            output[Columns.STATE_OUT] = qf_outs[Columns.STATE_OUT]
        # no backprop through target state
        if Columns.STATE_OUT in qf_target_next_outs:
            output[Columns.NEXT_STATE_OUT] = qf_target_next_outs[Columns.STATE_OUT]

        return output