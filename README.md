# IRFE-ECG

Code for the paper **"Separating Expert Retention from Autonomous Source
Inference in Raw-ECG-Replay-Free Continual ECG Deployment"**, accepted at
**IEEE BIBM 2026**.

Authors: Yufan Lu, Xinhui Liu, Chenyang Xu, Yuxi Zhou, and Hao Wang.

## Project overview

IRFE-ECG separates two questions in continual ECG deployment: whether frozen
source-specific experts retain their decisions, and whether an incoming ECG can
be routed to the correct expert without a source identifier. A frozen
ECGFounder backbone produces 1024-dimensional features. Each observed source
adds an isolated balanced-softmax linear expert, while centroid, kNN,
shrinkage-LDA, linear, and MLP routers infer the source from retained training
features. The primary autonomous method uses MLP top-2 validation-margin
fusion; equal probability averaging is reported as a control.

The paper uses seeds **42, 43, and 44** throughout. Versioned configurations in
`configs/` encode the reported settings explicitly rather than relying on
module defaults.

## Repository structure

```text
configs/                 Paper experiment configurations
scripts/run_suite.py     Cross-platform experiment launcher
src/data/                Processed-data loaders
src/models/              ECGFounder-compatible backbone and expert modules
src/trainer/             Expert, router, offline, and CL baseline implementations
src/evaluate/            Clinical and continual-learning metrics
src/scripts/             Data preparation and analysis utilities
tools/                   Paper table and figure generators
results/paper_tables/    Small aggregate, non-sample-level paper results
results/paper_figures/   Selected generated figures
docs/                    Protocol and code-availability documentation
tests/                   Unit tests for core routing and baseline logic
```

## Environment setup

Python 3.10 or 3.11 is recommended. Run commands from the repository root.
Multiline commands below use Bash continuations (`\`); in PowerShell, put
each command on one line. For pip:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For a CUDA-enabled Conda environment matching the paper software versions:

```bash
conda env create -f environment.yml
conda activate irfe-ecg
```

If CUDA 12.1 is unsuitable for the local driver, install the corresponding
PyTorch 2.4.0 build first, then install the remaining requirements.

## Data preparation

Raw ECG data, processed tensors, frozen feature caches, and pretrained weights
are not distributed in this repository. Obtain the relevant public records
from the [PhysioNet/Computing in Cardiology Challenge 2021 sources](https://physionet.org/content/challenge-2021/1.0.3/) and obtain
the single-lead checkpoint from the [official ECGFounder release](https://github.com/PKUDigitalHealth/ECGFounder), subject to
their respective terms.

The default local layout is:

```text
data/processed/
  Task1_CPSC_train.pt
  Task1_CPSC_test.pt
  Task2_PTBXL_train.pt
  Task2_PTBXL_test.pt
  Task3_Georgia_train.pt
  Task3_Georgia_test.pt
  Task4_Chapman_train.pt
  Task4_Chapman_test.pt
checkpoints/
  1_lead_ECGFounder.pth
feature_cache/            Generated locally; never committed
outputs/                  Generated locally; never committed
```

Utilities are provided for download inventory, preprocessing, split rebuilding,
and leakage auditing:

```bash
python -m scripts.download_physionet_challenge2021 --dest data/raw
python -m scripts.build_cinc_processed_from_raw \
  --raw-root data/raw --output-dir data/processed --seed 42
python -m scripts.audit_cinc_split_leakage --data-dir data/processed --fail-on-overlap
```

The exact source archives and preprocessing choices must comply with the data
providers' licenses. See [docs/DATA.md](docs/DATA.md) for the tensor contract,
grouping limitations, and the distinction between preparing compatible data
and reproducing the original paper splits.

## Running experiments

List commands without running training:

```bash
python scripts/run_suite.py --config configs/main_routing.json --dry-run
```

Run the paper suites with the shared tag `paper`, after checking a seed-42
smoke run as described in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md):

```bash
# Main table: pooled head, centroid, kNN, LDA, linear/MLP routers,
# MLP top-2 margin fusion, probability averaging, and source-aware oracle.
python scripts/run_suite.py --config configs/main_routing.json --tag paper

# Reverse and fixed-random domain orders.
python scripts/run_suite.py --config configs/order_robustness.json --tag paper

# Frozen-feature memory budgets: 1%, 5%, and 10%; 100% comes from the main run.
python scripts/run_suite.py --config configs/feature_memory.json --tag paper

# Shared head, full fine-tuning, EWC, LwF, SI, and 256/domain small replay.
python scripts/run_suite.py --config configs/shared_baselines.json --tag paper

# Matched offline independent-head reference.
python scripts/run_suite.py --config configs/offline_reference.json --tag paper

# Expert architecture and loss ablations.
python scripts/run_suite.py --config configs/expert_loss_ablation.json --tag paper
```

`main_routing.json`, `order_robustness.json`, and `feature_memory.json`
generate seed-specific frozen feature caches automatically when absent. The
caches are named `ecgfounder_final_seed{seed}_val0p15_{domain}_{split}.npz`.

## Experiment mapping

| Paper item | Implementation |
|---|---|
| Pooled frozen-feature head | `trainer.feature_pooled_head` |
| Centroid, kNN, LDA, linear, MLP routers | `trainer.feature_router_prototypes` |
| MLP top-2 validation-margin fusion | `domain_mlp_top2` |
| Equal probability-averaging control | `domain_mlp_probavg_top2` |
| Source-aware oracle | `oracle` setting in the router evaluation |
| Matched offline independent heads | `trainer.joint_offline_baseline` |
| Shared-parameter baselines | `trainer.shared_baselines` |
| Expert/loss ablations | `scripts.run_expert_loss_ablation` |

Top-2 margin fusion applies router-score softmax weights to each expert's
validation-threshold-centered decision margin. The probability-averaging
control instead equally averages positive-class probabilities and thresholds
the result at 0.5.

## Reproducing tables and figures

After all routing suites complete:

```bash
python tools/analyze_priority_router_results.py \
  --outputs-root outputs --stamp paper --output-dir outputs/router_analysis

python tools/generate_paper_router_artifacts.py \
  --outputs-root outputs --run-tag paper --output-dir outputs/paper_artifacts

python tools/generate_refined_ecg_figures.py \
  --outputs-root outputs --run-tag paper --output-dir outputs/paper_figures
```

Small aggregate reference tables and selected figures are retained under
`results/`. Sample-level predictions, checkpoints, logs, and full run folders
are intentionally excluded.

These generators require the local run outputs, not just the aggregate CSVs
in `results/`. The refined figure generator includes fixed paper values for
the domain-wise gap panel and fixed numeric caption text; those parts are
reference reproductions, not recomputed estimates for a new run.
See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for output mapping and limitations.

After installation, run the data-independent core tests with `python -m pytest -q`.

## Raw-ECG-replay-free, not memory-free

The primary protocol does not retain or replay raw historical ECG waveforms
when later sources arrive. It does retain frozen 1024-dimensional training
features for router fitting, together with the source-specific expert
parameters. Therefore, the protocol is **raw-ECG-replay-free but not
memory-free**. The small-replay baseline is a separate control that explicitly
stores model-input ECG tensors and is not part of the primary IRFE-ECG method.

## Code availability and data limitations

This repository releases code, versioned experiment configurations, aggregate
paper results, and instructions. Original ECG recordings, processed tensors,
pretrained weights, frozen feature caches, sample-level predictions, and other
large artifacts are not released. Users must obtain data from the relevant
public providers and construct compatible processed tensors or feature caches.
No claim is made that access to a public source overrides its license, data-use
agreement, or privacy requirements.

The current benchmark uses the paper's documented record-level grouped split
procedure. Users should review source metadata and construct patient-level
splits where reliable patient identifiers are available.

## Citation

```bibtex
@inproceedings{irfe_ecg2026,
  title     = {Separating Expert Retention from Autonomous Source Inference in Raw-ECG-Replay-Free Continual ECG Deployment},
  author    = {Lu, Yufan and Liu, Xinhui and Xu, Chenyang and Zhou, Yuxi and Wang, Hao},
  booktitle = {2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

DOI and page numbers will be added when available.

## License and attribution

The repository is released under the MIT License. Portions of the compatible
backbone implementation derive from ECGFounder and retain its MIT attribution;
see `THIRD_PARTY_NOTICES.md`.
