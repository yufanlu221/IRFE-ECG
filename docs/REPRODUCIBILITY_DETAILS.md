# Reproducibility Details

This document records settings in the current code and `configs/main_routing.json`
for seeds 42, 43, and 44. Items not represented in the current code or config
are marked explicitly. The expert checkpoint criterion below refers to the
primary router expert bank in `trainer.feature_router_prototypes`; other
baselines can use validation-calibrated epoch selection.

## 1. Domain MLP router

| Item | Verified setting |
|---|---|
| Input dimension | 1024 frozen ECGFounder features |
| Hidden dimension | 128 |
| Number of layers | One hidden layer; two affine layers (`1024 -> 128 -> K`) |
| LayerNorm | Affine `LayerNorm(1024)` immediately before the first linear layer |
| Activation | GELU |
| Dropout | 0.10, after GELU |
| Output dimension | `K = number of seen domains`; it grows from 1 to 4 across the stream |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001, inherited from `TRAIN_KWARGS` |
| Batch size | 512 retained features |
| Epochs | 50 fixed epochs |
| Training loss | Unweighted cross-entropy (`router_domain_balanced=false`) |
| Checkpoint criterion | Best validation route accuracy (`router_domain_select=accuracy`) |
| Early stopping | No early termination; all 50 epochs run and the best validation checkpoint is restored |
| Training protocol | Incremental seen-domain train features only; no future-domain router samples |
| Validation protocol | Full seen-domain validation features are used only to select the router checkpoint |

The stage with one seen domain uses the only available route directly. A new `K`-way classifier is fit at each stage using only the currently seen domains.

## 2. kNN router

| Item | Verified setting |
|---|---|
| `k` | 5 |
| Distance/similarity | Cosine similarity |
| Memory source | Retained train features from seen domains |
| Normalization | Both retained features and query features are L2-normalized |
| Decision | Domain vote among the top-5 neighbors; summed similarity is used only as a small tie-break term |

## 3. Centroid router

For each seen domain, the router computes the arithmetic mean of its retained training features. The centroid is L2-normalized. Query features are also L2-normalized, and routing uses maximum cosine similarity. No validation or test samples are used to compute centroids.

## 4. Shrinkage LDA

This is a custom NumPy implementation in `trainer/feature_router_prototypes.py`, not a scikit-learn estimator. It computes one mean vector per seen domain and a pooled within-domain covariance matrix. The covariance is shrunk toward its diagonal:

`Sigma_shrunk = (1 - 0.1) * Sigma + 0.1 * diag(Sigma)`.

An epsilon diagonal term is added and the pseudoinverse is used. Standard linear discriminant scores include empirical train-domain priors because `router_use_priors=true`. Only retained train features from seen domains are used. The primary run applies no z-score or whitening preprocessing (`router_preprocess=none`).

## 5. Feature-memory ablation

| Item | Verified behavior |
|---|---|
| Fractions | 1%, 5%, 10%, and 100% of each domain's train-feature rows |
| Sampling rule | `max(1, round(N_domain * fraction))`, sampled without replacement |
| Domain stratification | Yes in the operational sense that every domain is sampled independently at the same fraction |
| Class stratification | No |
| Seed handling | Each experiment seed samples separately |
| Random seed | Fixed as `SEED + 104729 * (domain_index + 1)` |
| Counted memory | Train features only |
| Validation features | Not counted as retained memory; full validation features are used for validation-only checkpoint/model selection of learned routers |
| Test features | Evaluation only |

## 6. Expert bank

| Item | Verified setting |
|---|---|
| Expert type | Independent linear head, `Linear(1024, 2)` |
| Parameters per expert | 2,050 |
| Four-domain expert parameters | 8,200 |
| Expert optimizer | AdamW, learning rate 0.0005, weight decay 0.0001 |
| Expert epochs | 30 fixed epochs with cosine learning-rate annealing |
| Expert checkpoint criterion | Validation Macro-F1 at threshold 0.5; best checkpoint restored after all epochs |
| Balanced Softmax counts | Binary class counts computed from that domain's training labels only |
| Threshold calibration | Per-domain validation-only search over 181 thresholds from 0.05 to 0.95 in increments of 0.005; maximize validation Macro-F1, breaking ties toward 0.5 |

Thresholds are calibrated after the best expert checkpoint is restored and are stored per domain and per seed. The held-out test split is not used for threshold selection.

## 7. Shared baselines

All shared baselines instantiate the same ECGFounder/Net1D architecture and load the same compatible pretrained checkpoint. They do not all use a frozen backbone:

- `linear_probe` freezes the backbone and trains only the recognized final classifier parameters.
- Domain-specific head-bank baselines freeze the backbone and train only the active head.
- `full_finetune`, EWC, LwF, `small_replay`, iCaRL-NCM, GDumb, and SI set all model parameters trainable.

The `small_replay` buffer stores `x` and `y` tensors copied from each domain's training `TensorDataset`. These are model-input ECG tensors, not frozen 1024-dimensional feature vectors. The configured budget is 256 training records per previous domain, selected approximately class-balanced with a fixed seed.

A partial memory/parameter comparison is quantifiable: the expert-bank parameters, MLP-router parameters, retained-feature counts, and replay record count are known. A fully equal-budget comparison is not established in the current code/config because shared methods update different parameter sets and the exact serialized replay memory includes input tensors, labels, and container overhead. The total trainable parameter count of each shared baseline and a measured byte-level replay-buffer footprint are `not found in current code/config`.
