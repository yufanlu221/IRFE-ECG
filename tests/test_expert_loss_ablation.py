from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_expert_loss_ablation import (
    VARIANTS,
    balanced_softmax_loss,
    class_counts_and_weights,
    select_validation_bank,
)


def test_train_only_class_weights_match_binary_inverse_frequency_formula() -> None:
    labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    counts, weights = class_counts_and_weights(labels, max_class_weight=10.0)
    np.testing.assert_array_equal(counts, np.array([2.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(weights, np.array([1.5, 0.75], dtype=np.float32))


def test_balanced_softmax_matches_repository_log_count_adjustment() -> None:
    logits = torch.tensor([[0.2, -0.1], [0.3, 0.7]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    counts = torch.tensor([2.0, 8.0], dtype=torch.float32)
    actual = balanced_softmax_loss(logits, labels, counts)
    expected = F.cross_entropy(logits + counts.log(), labels)
    torch.testing.assert_close(actual, expected)


def test_validation_bank_ignores_better_test_f1() -> None:
    records = []
    for seed in (42,):
        for domain in ("cpsc", "ptbxl", "georgia", "chapman"):
            for index, variant in enumerate(VARIANTS):
                records.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "domain": domain,
                        "val_macro_f1": 0.90 - 0.05 * index,
                        # The worst validation candidate has the best test F1.
                        "f1": 0.60 + 0.15 * index,
                    }
                )
    selected = select_validation_bank(records)
    assert len(selected) == 4
    assert {row["selected_variant"] for row in selected} == {VARIANTS[0]}
    assert {row["f1"] for row in selected} == {0.60}
