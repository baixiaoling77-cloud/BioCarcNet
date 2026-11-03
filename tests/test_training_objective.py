import pytest
import torch

from biocarcnet.training.engine import (
    _masked_training_loss,
    _pairwise_ranking_loss,
    calculate_binary_metrics,
)


def test_contrastive_mask_supports_multiple_tasks():
    logits = torch.zeros((3, 2), requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    mask = torch.ones_like(target)
    conflict = torch.zeros(3)

    loss = _masked_training_loss(
        logits, target, mask, conflict, ["CC", "CC", "CO"]
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_pairwise_ranking_loss_rewards_correct_ordering():
    target = torch.tensor([[1.0], [0.0]])
    mask = torch.ones_like(target)

    correctly_ordered = _pairwise_ranking_loss(
        torch.tensor([[2.0], [-2.0]]), target, mask
    )
    incorrectly_ordered = _pairwise_ranking_loss(
        torch.tensor([[-2.0], [2.0]]), target, mask
    )

    assert correctly_ordered < incorrectly_ordered


def test_pairwise_ranking_loss_is_zero_without_eligible_pair():
    logits = torch.tensor([[1.0], [2.0]], requires_grad=True)
    target = torch.ones_like(logits)
    loss = _pairwise_ranking_loss(logits, target, torch.ones_like(target))
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_manuscript_metrics_are_calculated_at_threshold_point_five():
    metrics = calculate_binary_metrics(
        predictions=[0.1, 0.8, 0.7, 0.9],
        labels=[0, 0, 1, 1],
    )

    assert metrics["n"] == 4
    assert metrics["RA"] == pytest.approx(0.75)
    assert metrics["RBA"] == pytest.approx(0.75)
    assert metrics["RP"] == pytest.approx(2 / 3)
    assert metrics["RTP"] == pytest.approx(1.0)
    assert metrics["RTN"] == pytest.approx(0.5)
    assert metrics["F1"] == pytest.approx(0.8)
    assert metrics["AROC"] == pytest.approx(0.75)
    assert 0.0 <= metrics["APR"] <= 1.0
