import numpy as np
import torch
import torch.nn as nn

import trainer.feature_router_prototypes as router


def test_router_mode_suffixes_and_memory_indices_are_deterministic(monkeypatch):
    assert router._base_router_mode("domain_mlp_top2") == "domain_mlp"
    assert router._base_router_mode("domain_mlp_probavg_top2") == "domain_mlp"
    assert router._is_probability_average_mode("domain_mlp_probavg_top2")

    monkeypatch.setattr(router, "ROUTER_MEMORY_FRACTION", 0.1)
    first = router._router_memory_indices("cpsc", 100)
    second = router._router_memory_indices("cpsc", 100)
    assert first.shape == (10,)
    np.testing.assert_array_equal(first, second)


def test_knn_router_scores_vote_for_nearest_domain(monkeypatch):
    monkeypatch.setattr(router, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(router, "ROUTER_KNN_K", 1)
    train_x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    train_domains = np.asarray([0, 1], dtype=np.int64)
    query = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    scores = router._knn_router_scores(query, train_x, train_domains, 2)
    np.testing.assert_array_equal(scores.argmax(axis=1), np.asarray([0, 1]))


class _ConstantHead(nn.Module):
    def __init__(self, positive_probability: float) -> None:
        super().__init__()
        logit = float(np.log(positive_probability / (1.0 - positive_probability)))
        self.register_buffer("positive_logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive = self.positive_logit.expand(x.shape[0])
        return torch.stack([torch.zeros_like(positive), positive], dim=1)


def test_probability_average_is_unweighted_top2(monkeypatch):
    monkeypatch.setattr(router, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(router, "ROUTER_TOPK", 2)
    x = np.zeros((3, 2), dtype=np.float32)
    scores = np.asarray([[2.0, 1.0], [1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    heads = {"a": _ConstantHead(0.8), "b": _ConstantHead(0.4)}
    probs, preds = router._routed_topk_probability_average(
        x, ["a", "b"], scores, heads
    )
    np.testing.assert_allclose(probs, np.full(3, 0.6), atol=1e-6)
    np.testing.assert_array_equal(preds, np.ones(3, dtype=np.int64))
