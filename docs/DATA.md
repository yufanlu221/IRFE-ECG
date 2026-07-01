# Data contract

The training code expects four binary, single-lead ECG domains represented as
PyTorch tensor files. Each domain has a train and test file named as listed in
the root README. The loader accepts either a `TensorDataset`, a dictionary with
input and target tensors, or the compatible tuple structure documented in
`data.loaders.cinc_dataset`.

Validation data is derived only from the training split with a fixed ratio of
0.15 and the active experiment seed. Test data is held out until evaluation.
The paper seeds are 42, 43, and 44.

The preprocessing utility defaults to lead I, 10 seconds, 500 Hz, no amplitude
normalization, grouped splitting when source metadata permits it, and seed 42
for the initial train/test construction. Verify the generated split manifest
and run the leakage audit before training.

Neither raw recordings nor processed tensors may be committed to this
repository. Place local files under `data/` or `datasets/`, both of which are
ignored by Git.
