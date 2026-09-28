# Data contract

The training code expects four binary, single-lead ECG domains represented as
PyTorch tensor files. Each domain has a train and test file named as listed in
the root README. The loader expects a dictionary containing `x` (shape
`N x 1 x L`, converted to float32) and `y` (shape `N`, converted to int64),
not a serialized `TensorDataset` or tuple. Labels are 0 for Normal (diagnoses
contain only SNOMED code 426783006) and 1 for Abnormal. Include
`label_task="normal_abnormal"` and
`label_description="binary ECG abnormality screening: 0=Normal, 1=Abnormal"`.
The default strict loader rejects missing or inconsistent label metadata.

Validation data is derived only from the training split with a fixed ratio of
0.15 and the active experiment seed. Test data is held out until evaluation.
The paper seeds are 42, 43, and 44.

The preprocessing utility defaults to lead I, 10 seconds, 500 Hz, no amplitude
normalization, grouped splitting when source metadata permits it, and seed 42
for the initial train/test construction. Verify the generated split manifest
and run the leakage audit before training.

The downloader writes `data/raw/training/{cpsc_2018,ptb-xl,georgia,chapman_shaoxing}/`;
pass `data/raw` as the preprocessing `--raw-root`. The default initial
train/test ratio is 0.8. The 0.15 validation ratio is applied to groups within
the training split, so the exact fraction of records can differ.

The default `--group-by auto` uses patient identifiers present in headers and
otherwise groups by exact waveform hash. `--group-by patient` falls back to
record IDs when a header has no patient identifier. Neither fallback proves
patient independence. The leakage audit detects exact waveform overlap, not
all repeated patients. Use verified patient metadata whenever available;
never infer patient identities from demographics, filenames, or waveforms.
Retain `group_id`, `group_kind`, record metadata, and the generated split report.

These commands prepare compatible inputs; the release does not include the
original per-record split manifests or data, so exact agreement with the
paper's record-level grouped benchmark cannot be established from the
aggregate tables alone. A new patient-level split is a separate evaluation
and must not replace the official paper results.

Neither raw recordings nor processed tensors may be committed to this
repository. Place local files under `data/` or `datasets/`, both of which are
ignored by Git.
