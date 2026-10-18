import numpy as np
import pytest
import torch

from sapiens_llm import grpo, metrics
from sapiens_llm.topology import TOPOLOGIES, Topology
from sapiens_llm.train import DEFAULTS


def test_group_advantages_standardise_within_each_prompt():
    adv = grpo.group_advantages([1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    assert adv[0] > 0 > adv[1] and abs(adv[:3].sum()) < 1e-5
    assert np.allclose(adv[3:], 0)          # all equal rewards: no signal


def test_collate_marks_completion_targets():
    ids, att, mask = grpo.collate([dict(prompt_ids=[1, 2, 3], completion_ids=[4, 5]),
                                   dict(prompt_ids=[1], completion_ids=[6, 7, 8])], pad_id=0)
    assert ids.tolist() == [[1, 2, 3, 4, 5], [1, 6, 7, 8, 0]]
    # targets are ids[:, 1:]; completion targets are exactly the completion tokens
    targets = ids[:, 1:]
    assert targets[0][mask[0] > 0].tolist() == [4, 5]
    assert targets[1][mask[1] > 0].tolist() == [6, 7, 8]


def test_policy_loss_weights_received_tokens_by_truncated_ratio():
    lp = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]])).requires_grad_(True)
    behaviour = torch.log(torch.tensor([[0.5, 0.5], [0.25, 1.0]]))
    adv, mask = torch.tensor([1.0, 1.0]), torch.ones(2, 2)
    grpo.policy_loss(lp, behaviour, adv, mask, torch.tensor([False, True]), clip=1.0).backward()
    # own row: weight 1; received row: min(0.5/0.25, 1) = 1 and min(0.5/1.0, 1) = 0.5
    assert torch.allclose(lp.grad, -torch.tensor([[1.0, 1.0], [1.0, 0.5]]))


def test_pass_at_k_and_group():
    assert metrics.pass_at_k(4, 0, 2) == 0.0 and metrics.pass_at_k(4, 4, 1) == 1.0
    assert abs(metrics.pass_at_k(4, 1, 2) - 0.5) < 1e-9
    correct = np.zeros((2, 1, 4), bool)
    correct[0, 0, 0] = correct[1, 0, 0] = True       # each learner solves 1 of 4
    assert abs(metrics.group_pass_at_k(correct, 1) - (1 - 0.75 ** 2)) < 1e-9


def test_last_boxed_and_conformity():
    assert metrics.last_boxed(r"so \boxed{\frac{1}{2}} and finally \boxed{3}") == "3"
    assert metrics.last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert metrics.conformity([[["3", "3", "4"]], [["3", None, "3"]]]) == 1.0
    assert metrics.conformity([[["3"]], [["4"]]]) == 0.0


@pytest.mark.parametrize("kind", TOPOLOGIES)
def test_topologies_symmetric_no_self_loops(kind):
    topo = Topology(dict(DEFAULTS, TOPOLOGY=kind, NUM_LEARNERS=6, P_VISIT=0.5), np.random.default_rng(0))
    for it in range(30):
        adj = topo.adjacency(it)
        assert (adj == adj.T).all() and not adj.diagonal().any()
        topo.step(it)
