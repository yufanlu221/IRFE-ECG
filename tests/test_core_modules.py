import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.loaders.cinc_dataset import (
    LABEL_DESCRIPTION,
    LABEL_TASK,
    _validate_label_metadata,
    get_single_loader,
)
from evaluate.clinical_metrics import binary_clinical_metrics
from models.adapter.domain_adapter import AdapterCLModel
from models.head.fast_slow_head import FastSlowHead
from scripts.build_cinc_processed_from_raw import Record, label_from_dx, split_records
from scripts.patch_cinc_metadata import patch_file
from trainer.continual_cl import (
    calibrate_threshold,
    collect_labels_probs_routed,
    compute_class_weights_from_labels,
    compute_forgetting,
    evaluate,
    evaluate_routed,
    fit_learned_linear_router,
    make_val_split,
    route_features_by_prototype,
    route_features_by_confidence_margin,
    router_variant_specs,
    restore_domain_state,
    snapshot_domain_state,
    summarize_router_routes,
)


def make_test_dir(name: str) -> Path:
    root = Path("tests") / "audit_tmp" / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


class TinyBackbone(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.stage_list = nn.ModuleList([nn.Conv1d(1, channels, kernel_size=3, padding=1)])
        self.dense = nn.Linear(channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stage_list[0](x)
        return self.dense(feat.mean(dim=-1))


def test_fast_slow_head_updates_and_merges() -> None:
    head = FastSlowHead(embed_dim=4, num_classes=2, ema_beta=0.5, fast_alpha=0.25)
    x = torch.randn(3, 4)

    logits_fast = head(x)
    assert logits_fast.shape == (3, 2)
    assert not head.slow_initialized

    head.update_slow()
    assert head.slow_initialized

    head.reset_slow()
    assert head.slow_initialized

    head.eval()
    logits_merged = head(x, use_merge=True)
    assert logits_merged.shape == (3, 2)


def test_fast_slow_head_rejects_merged_training_forward() -> None:
    head = FastSlowHead(embed_dim=4, num_classes=2)
    head.update_slow()
    x = torch.randn(3, 4)
    try:
        head(x, use_merge=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("use_merge=True should be rejected in training mode")


def test_ema_interpolates_not_copies() -> None:
    head = FastSlowHead(embed_dim=4, num_classes=2, ema_beta=0.5, fast_alpha=0.5)
    head.update_slow()
    w_init = head.W_slow_weight.clone()

    with torch.no_grad():
        head.W_fast.weight.fill_(10.0)

    head.update_slow()
    expected = 0.5 * w_init + 0.5 * 10.0
    assert not torch.allclose(head.W_slow_weight, torch.full_like(head.W_slow_weight, 10.0))
    assert not torch.allclose(head.W_slow_weight, w_init)
    assert torch.allclose(head.W_slow_weight, expected, atol=1e-5)


def test_reinitialize_slow_uses_fast_only_with_warning() -> None:
    head = FastSlowHead(embed_dim=4, num_classes=2, ema_beta=0.5)
    head.update_slow()
    head.reinitialize_slow()
    assert not head.slow_initialized

    head.eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logits = head(torch.randn(2, 4), use_merge=True)

    assert logits.shape == (2, 2)
    assert any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_adapter_model_freezes_inactive_domains() -> None:
    model = AdapterCLModel(
        backbone=TinyBackbone(channels=4),
        domain_names=["cpsc", "ptbxl"],
        embed_dim=4,
        bottleneck=2,
        num_classes=2,
        hook_stage=0,
    )
    x = torch.randn(2, 1, 16)

    logits = model(x, domain="cpsc")
    assert logits.shape == (2, 2)
    model.update_slow("cpsc")

    model.set_domain("ptbxl")
    model.set_head_hparams("ptbxl", ema_beta=0.8, fast_alpha=0.4)
    assert model.heads["ptbxl"].ema_beta == 0.8
    assert model.heads["ptbxl"].fast_alpha == 0.4

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("adapters.ptbxl") for name in trainable)
    assert any(name.startswith("heads.ptbxl") for name in trainable)
    assert not any(name.startswith("adapters.cpsc") for name in trainable)
    assert not any(name.startswith("heads.cpsc") for name in trainable)


def test_gradients_do_not_reach_backbone() -> None:
    model = AdapterCLModel(
        backbone=TinyBackbone(channels=4),
        domain_names=["cpsc"],
        embed_dim=4,
        bottleneck=2,
        num_classes=2,
        hook_stage=0,
    )
    x = torch.randn(2, 1, 16)

    model.train()
    loss = model(x, domain="cpsc").sum()
    loss.backward()

    for name, param in model.backbone.named_parameters():
        assert param.grad is None, f"backbone param {name} has gradient"
    assert any(
        param.grad is not None for param in model.adapters["cpsc"].parameters()
    ), "adapter received no gradients"
    assert any(
        param.grad is not None for param in model.heads["cpsc"].parameters()
    ), "head received no gradients"


def test_forward_raises_on_wrong_domain_in_training() -> None:
    model = AdapterCLModel(
        backbone=TinyBackbone(channels=4),
        domain_names=["cpsc", "ptbxl"],
        embed_dim=4,
        bottleneck=2,
        num_classes=2,
        hook_stage=0,
    )
    model.train()
    model.set_domain("ptbxl")

    try:
        model(torch.randn(2, 1, 16), domain="cpsc")
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong-domain training forward should have failed")


def test_forward_features_matches_forward_after_feature_extraction() -> None:
    model = AdapterCLModel(
        backbone=TinyBackbone(channels=4),
        domain_names=["cpsc"],
        embed_dim=4,
        bottleneck=2,
        num_classes=2,
        hook_stage=0,
    )
    x = torch.randn(3, 1, 16)
    model.eval()

    logits_from_x = model(x, domain="cpsc")
    features = model.extract_features(x)
    logits_from_features = model.forward_features(features, domain="cpsc")

    assert torch.allclose(logits_from_x, logits_from_features, atol=1e-6)


def test_binary_clinical_metrics_confusion_fields() -> None:
    metrics = binary_clinical_metrics(
        labels=[0, 0, 1, 1],
        probs=[[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2]],
    )
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5


def test_macro_f1_equals_average_of_per_class_f1() -> None:
    metrics = binary_clinical_metrics(
        labels=[0, 0, 1, 1],
        probs=[0.1, 0.6, 0.7, 0.2],
    )
    expected = (metrics["f1_negative"] + metrics["f1_positive"]) / 2.0
    assert abs(metrics["f1"] - expected) < 1e-6


def test_strict_label_metadata_rejects_missing_metadata() -> None:
    try:
        _validate_label_metadata({}, Path("Task1_CPSC_train.pt"), strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("missing label metadata should have failed")


def test_non_strict_label_metadata_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_label_metadata({}, Path("Task1_CPSC_train.pt"), strict=False)

    assert any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_get_single_loader_allows_num_workers_override() -> None:
    tmp = make_test_dir("loader")
    try:
        pt_path = tmp / "Task1_CPSC_train.pt"
        torch.save(
            {
                "x": torch.randn(3, 1, 8),
                "y": torch.tensor([0, 1, 1]),
                "label_task": LABEL_TASK,
                "label_description": LABEL_DESCRIPTION,
            },
            pt_path,
        )
        loader = get_single_loader(str(tmp), "cpsc", num_workers=1)
        assert loader.num_workers == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_label_from_dx_requires_nonempty_codes() -> None:
    assert label_from_dx(["426783006"]) == 0
    assert label_from_dx(["426783006", "164889003"]) == 1
    try:
        label_from_dx([])
    except ValueError:
        pass
    else:
        raise AssertionError("missing Dx codes should not be silently labeled")


def test_split_records_keeps_duplicate_waveforms_together() -> None:
    records = []
    for idx, (group_id, label) in enumerate(
        [
            ("dup_waveform", 0),
            ("dup_waveform", 0),
            ("normal_unique", 0),
            ("abnormal_a", 1),
            ("abnormal_b", 1),
            ("abnormal_c", 1),
        ]
    ):
        records.append(
            Record(
                x=torch.zeros(1, 8).numpy(),
                y=label,
                record_id=f"r{idx}",
                source_database="test",
                source_path=f"r{idx}.hea",
                dx_codes=["426783006"] if label == 0 else ["164889003"],
                age=None,
                sex=None,
                fs=500,
                n_samples=8,
                group_id=group_id,
                group_kind="waveform_hash_fallback",
                waveform_hash=group_id,
            )
        )

    train_indices, test_indices = split_records(records, train_ratio=0.5, seed=7)
    train_groups = {records[idx].group_id for idx in train_indices}
    test_groups = {records[idx].group_id for idx in test_indices}
    assert train_groups.isdisjoint(test_groups)


def test_compute_forgetting_excludes_final_task_from_bwt() -> None:
    matrix = torch.tensor(
        [
            [0.8, float("nan"), float("nan")],
            [0.7, 0.6, float("nan")],
            [0.6, 0.5, 0.4],
        ],
        dtype=torch.float32,
    ).numpy()
    result = compute_forgetting(matrix, ["a", "b", "c"])
    assert result["bwt_num_tasks"] == 2
    assert len(result["bwt"]) == 2
    assert abs(result["mean_bwt"] - (-0.15)) < 1e-6


def test_make_val_split_keeps_groups_disjoint() -> None:
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    groups = ["n0", "n0", "n1", "n1", "p0", "p0", "p1", "p1"]

    train_idx, val_idx, info = make_val_split(
        labels, groups=groups, val_ratio=0.5, seed=3, group_key="group_id"
    )

    train_groups = {groups[idx] for idx in train_idx}
    val_groups = {groups[idx] for idx in val_idx}
    assert train_groups.isdisjoint(val_groups)
    assert info["group_overlap_count"] == 0
    assert info["val_n"] > 0


def test_class_weights_use_train_subset_labels_only() -> None:
    train_subset_labels = torch.tensor([0, 1, 1, 1])
    weights = compute_class_weights_from_labels(train_subset_labels)

    assert torch.allclose(weights, torch.tensor([2.0, 2.0 / 3.0]), atol=1e-6)


def test_calibrate_threshold_uses_validation_macro_f1() -> None:
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    probs = torch.tensor([0.1, 0.2, 0.4, 0.9]).numpy()

    threshold, metrics = calibrate_threshold(
        labels, probs, thresholds=torch.tensor([0.3, 0.5]).numpy()
    )

    assert abs(threshold - 0.3) < 1e-6
    assert metrics["f1"] > binary_clinical_metrics(labels, probs, threshold=0.5)["f1"]


def test_snapshot_restore_domain_state_roundtrip() -> None:
    model = AdapterCLModel(
        backbone=TinyBackbone(channels=4),
        domain_names=["cpsc"],
        embed_dim=4,
        bottleneck=2,
        num_classes=2,
        hook_stage=0,
    )
    state = snapshot_domain_state(model, "cpsc")

    with torch.no_grad():
        for param in model.adapters["cpsc"].parameters():
            param.add_(1.0)
        model.heads["cpsc"].W_fast.bias.add_(1.0)

    restore_domain_state(model, "cpsc", state)
    for name, tensor in model.adapters["cpsc"].state_dict().items():
        assert torch.allclose(tensor, state["adapter"][name])
    for name, tensor in model.heads["cpsc"].state_dict().items():
        assert torch.allclose(tensor, state["head"][name])


def test_evaluate_respects_threshold_and_use_merge() -> None:
    class MergeSwitchModel(nn.Module):
        def forward(self, x, domain=None, use_merge=False):
            if use_merge:
                return torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
            return torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    loader = DataLoader(
        TensorDataset(torch.zeros(3, 1, 4), torch.tensor([1, 0, 1])),
        batch_size=3,
    )

    merged = evaluate(MergeSwitchModel(), loader, domain="cpsc", use_merge=True)
    fast = evaluate(MergeSwitchModel(), loader, domain="cpsc", use_merge=False)
    assert merged["f1"] > fast["f1"]


def test_route_features_by_prototype_selects_nearest_domain() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    prototypes = {
        "cpsc": {"centroid": torch.tensor([1.0, 0.0])},
        "ptbxl": {"centroid": torch.tensor([0.0, 1.0])},
    }

    routes, scores = route_features_by_prototype(
        features, prototypes, ["cpsc", "ptbxl"], distance="cosine"
    )

    assert routes == ["cpsc", "ptbxl", "cpsc"]
    assert scores.shape == (3, 2)


def test_route_features_by_prototype_supports_mahalanobis_variance() -> None:
    features = torch.tensor([[0.0, 2.0]])
    prototypes = {
        "wide": {
            "centroid": torch.tensor([0.0, 0.0]),
            "variance": torch.tensor([1.0, 100.0]),
        },
        "tight": {
            "centroid": torch.tensor([0.0, 1.8]),
            "variance": torch.tensor([1.0, 0.01]),
        },
    }

    routes, scores = route_features_by_prototype(
        features, prototypes, ["wide", "tight"], distance="mahalanobis"
    )

    assert routes == ["wide"]
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]


def test_route_features_by_prototype_supports_class_conditional_centroids() -> None:
    features = torch.tensor([[9.8, 0.0], [0.0, 5.1]])
    prototypes = {
        "cpsc": {
            "centroid": torch.tensor([0.0, 0.0]),
            "class_centroids": {
                "0": torch.tensor([0.0, 0.0]),
                "1": torch.tensor([10.0, 0.0]),
            },
        },
        "ptbxl": {
            "centroid": torch.tensor([0.0, 5.0]),
            "class_centroids": {
                "0": torch.tensor([0.0, 5.0]),
                "1": torch.tensor([0.0, 6.0]),
            },
        },
    }

    routes, scores = route_features_by_prototype(
        features,
        prototypes,
        ["cpsc", "ptbxl"],
        distance="euclidean",
        prototype="class_conditional",
    )

    assert routes == ["cpsc", "ptbxl"]
    assert scores.shape == (2, 2)


def test_route_features_by_prototype_supports_subspace_residual() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = {
        "x_axis": {
            "mean": torch.tensor([0.0, 0.0]),
            "basis": torch.tensor([[1.0], [0.0]]),
        },
        "y_axis": {
            "mean": torch.tensor([0.0, 0.0]),
            "basis": torch.tensor([[0.0], [1.0]]),
        },
    }

    routes, scores = route_features_by_prototype(
        features,
        prototypes,
        ["x_axis", "y_axis"],
        distance="residual",
        prototype="subspace",
    )

    assert routes == ["x_axis", "y_axis"]
    assert scores.shape == (2, 2)


def test_fit_learned_linear_router_routes_separable_features() -> None:
    memory = {
        "cpsc": torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.9, 0.0]]),
        "chapman": torch.tensor([[0.0, 1.0], [0.1, 1.0], [0.0, 0.9]]),
    }
    router = fit_learned_linear_router(memory, ["cpsc", "chapman"])

    routes, scores = route_features_by_prototype(
        torch.tensor([[0.8, 0.0], [0.0, 0.8]]),
        {"__router__": router},
        ["cpsc", "chapman"],
        distance="linear",
        prototype="learned_linear",
    )

    assert routes == ["cpsc", "chapman"]
    assert scores.shape == (2, 2)
    assert router["train_acc"] >= 0.99


def test_route_features_by_confidence_margin_selects_most_confident_expert() -> None:
    class ConfidenceModel(nn.Module):
        def forward_features(self, features, domain=None, use_merge=False):
            values = features[:, 0] if domain == "cpsc" else features[:, 1]
            return torch.stack([torch.zeros_like(values), values], dim=1)

    features = torch.tensor([[3.0, 0.2], [0.1, 2.0]])
    routes, scores, logits = route_features_by_confidence_margin(
        ConfidenceModel(),
        features,
        ["cpsc", "ptbxl"],
    )

    assert routes == ["cpsc", "ptbxl"]
    assert scores.shape == (2, 2)
    assert logits.shape == (2, 2)


def test_summarize_router_routes_counts_and_distribution() -> None:
    summary = summarize_router_routes(
        ["cpsc", "ptbxl", "ptbxl"],
        ["cpsc", "ptbxl", "georgia"],
    )

    assert summary["total"] == 3
    assert summary["counts"] == {"cpsc": 1, "ptbxl": 2, "georgia": 0}
    assert abs(summary["distribution"]["ptbxl"] - 2.0 / 3.0) < 1e-6


def test_router_variant_specs_parses_default_sweep_once() -> None:
    specs = router_variant_specs(
        sweep="default",
        primary_prototype="domain",
        primary_distance="cosine",
    )

    names = [name for name, _, _ in specs]
    assert names == [
        "domain_cosine",
        "domain_euclidean",
        "domain_mahalanobis",
        "class_conditional_cosine",
        "class_conditional_mahalanobis",
        "confidence_margin",
    ]


def test_router_variant_specs_parses_extended_sweep() -> None:
    specs = router_variant_specs(
        sweep="extended",
        primary_prototype="domain",
        primary_distance="cosine",
    )

    names = [name for name, _, _ in specs]
    assert "domain_cosine_top2" in names
    assert "domain_cosine_uncertain_top2" in names
    assert "subspace_residual" in names
    assert "learned_linear" in names
    assert "learned_linear_uncertain_top2" in names


def test_evaluate_routed_uses_router_chosen_expert_thresholds() -> None:
    class RoutedToyModel(nn.Module):
        def eval(self):
            return self

        def extract_features(self, x):
            return x.squeeze(1)

        def forward_features(self, feat, domain=None, use_merge=False):
            if domain == "cpsc":
                return torch.stack(
                    [
                        torch.zeros(len(feat), device=feat.device),
                        torch.full((len(feat),), 4.0, device=feat.device),
                    ],
                    dim=1,
                )
            return torch.stack(
                [
                    torch.full((len(feat),), 4.0, device=feat.device),
                    torch.zeros(len(feat), device=feat.device),
                ],
                dim=1,
            )

    loader = DataLoader(
        TensorDataset(
            torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            torch.tensor([1, 0]),
        ),
        batch_size=2,
    )
    prototypes = {
        "cpsc": {"centroid": torch.tensor([1.0, 0.0])},
        "ptbxl": {"centroid": torch.tensor([0.0, 1.0])},
    }

    metrics = evaluate_routed(
        RoutedToyModel(),
        loader,
        eval_domain="cpsc",
        candidate_domains=["cpsc", "ptbxl"],
        prototypes=prototypes,
        thresholds={"cpsc": 0.5, "ptbxl": 0.5},
    )

    assert metrics["f1"] == 1.0
    assert metrics["router_acc"] == 0.5


def test_collect_labels_probs_routed_domain_top2_blends_logits() -> None:
    class Top2ToyModel(nn.Module):
        def eval(self):
            return self

        def extract_features(self, x):
            return x.squeeze(1)

        def forward_features(self, feat, domain=None, use_merge=False):
            if domain == "cpsc":
                return torch.stack(
                    [
                        torch.zeros(len(feat), device=feat.device),
                        torch.full((len(feat),), 4.0, device=feat.device),
                    ],
                    dim=1,
                )
            return torch.stack(
                [
                    torch.full((len(feat),), 4.0, device=feat.device),
                    torch.zeros(len(feat), device=feat.device),
                ],
                dim=1,
            )

    loader = DataLoader(
        TensorDataset(torch.tensor([[[1.0, 0.0]]]), torch.tensor([1])),
        batch_size=1,
    )
    prototypes = {
        "cpsc": {"centroid": torch.tensor([1.0, 0.0])},
        "ptbxl": {"centroid": torch.tensor([0.8, 0.2])},
    }

    labels, probs, preds, routes, scores = collect_labels_probs_routed(
        Top2ToyModel(),
        loader,
        eval_domain="cpsc",
        candidate_domains=["cpsc", "ptbxl"],
        prototypes=prototypes,
        thresholds={"cpsc": 0.5, "ptbxl": 0.5},
        prototype="domain_top2",
        distance="cosine",
    )

    assert labels.tolist() == [1]
    assert routes == ["cpsc"]
    assert scores.shape == (1, 2)
    assert probs[0] > 0.5
    assert preds.tolist() == [1]


def test_evaluate_routed_uncertain_top2_reports_fraction() -> None:
    class UncertainTop2ToyModel(nn.Module):
        def eval(self):
            return self

        def extract_features(self, x):
            return x.squeeze(1)

        def forward_features(self, feat, domain=None, use_merge=False):
            if domain == "cpsc":
                values = torch.full((len(feat),), 4.0, device=feat.device)
            else:
                values = torch.full((len(feat),), 3.5, device=feat.device)
            return torch.stack([torch.zeros_like(values), values], dim=1)

    loader = DataLoader(
        TensorDataset(torch.tensor([[[1.0, 0.0]]]), torch.tensor([1])),
        batch_size=1,
    )
    prototypes = {
        "cpsc": {"centroid": torch.tensor([1.0, 0.0])},
        "ptbxl": {"centroid": torch.tensor([0.99, 0.01])},
    }

    metrics = evaluate_routed(
        UncertainTop2ToyModel(),
        loader,
        eval_domain="cpsc",
        candidate_domains=["cpsc", "ptbxl"],
        prototypes=prototypes,
        thresholds={"cpsc": 0.5, "ptbxl": 0.5},
        prototype="domain_uncertain_top2",
        distance="cosine",
    )

    assert metrics["router_acc"] == 1.0
    assert metrics["router_uncertain_fraction"] == 1.0


def test_patch_cinc_metadata_adds_missing_fields() -> None:
    tmp = make_test_dir("metadata")
    try:
        pt_path = tmp / "Task1_CPSC_train.pt"
        torch.save({"x": torch.randn(2, 1, 8), "y": torch.tensor([0, 1])}, pt_path)

        dry_run = patch_file(pt_path, write=False, force=False)
        assert dry_run["status"] == "would_patch"

        patched = patch_file(pt_path, write=True, force=False)
        assert patched["status"] == "patched"

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert data["label_task"] == LABEL_TASK
        assert data["label_description"] == LABEL_DESCRIPTION
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
