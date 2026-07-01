# Experiment guide

All formal suites use seeds 42, 43, and 44. `scripts/run_suite.py` rejects a
different seed set for versioned paper configurations.

## Main routing

`configs/main_routing.json` first generates frozen ECGFounder features when
needed. It trains one balanced-softmax linear expert per source and evaluates a
pooled head, centroid, kNN, shrinkage-LDA, linear, and MLP routers. The same run
also reports hard MLP routing, top-2 validation-margin fusion, equal
probability averaging, and source-aware oracle selection.

## Robustness and memory

`configs/order_robustness.json` evaluates reverse and fixed-random source
orders. `configs/feature_memory.json` evaluates 1%, 5%, and 10% retained
feature fractions; the main configuration provides the 100% point.

## References and ablations

`configs/shared_baselines.json` covers the shared linear head, full
fine-tuning, EWC, LwF, SI, and small replay. `configs/offline_reference.json`
trains matched independent heads with all source training sets available up
front. `configs/expert_loss_ablation.json` compares linear balanced softmax,
linear weighted cross-entropy, and a residual feature-adapter expert.

Generated checkpoints, cached features, sample-level predictions, and logs are
written below ignored directories and are not publication artifacts.
