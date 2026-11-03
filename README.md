# BioCarcNet

BioCarcNet is a biological condition-aware model for rodent carcinogenicity
prediction. It integrates atom-graph, bond-relation-graph, and molecular
fingerprint representations with species and sex information.

## Repository structure

```text
BioCarcNet-public/
├── data/                 Modeling dataset and data documentation
├── examples/             Example prediction input
├── scripts/              Command-line wrappers
├── src/biocarcnet/       Reusable Python package
├── tests/                Unit and smoke tests
├── pyproject.toml         Package and command definitions
└── requirements.txt      Runtime dependencies
```

## Installation

Python 3.10 or later is recommended. Create an isolated environment and run:

```bash
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
pytest
```

For Excel prediction input, install the optional I/O dependencies:

```bash
python -m pip install -e ".[io]"
```

## Modeling dataset

The modeling dataset is provided at `data/biocarcnet.csv`. It contains 5,847
chemical-species-sex records for 2,314 unique chemicals and uses the following
column order:

```text
smiles,label,species,sex
```

- `label`: `0=negative`, `1=positive`
- `species`: `0=rat`, `1=mouse`
- `sex`: `0=male`, `1=female`

The independent ISSCAN dataset is also provided in `data/`. See
`data/README.md` for dataset summaries and file integrity information.

## Data splitting

The code follows the terminology used in the manuscript. Both supported
strategies produce an 8:1:1 **training/test/validation** split:

- `record`: records are jointly stratified by species, sex, and label. The same
  chemical can occur in different subsets under different biological
  conditions.
- `chemical`: all records with the same canonical isomeric SMILES are assigned
  to one subset, preventing chemical overlap among subsets.

The training set updates model parameters, the test set is used for model
selection, and the validation set is evaluated only after model selection.

## Model inputs

- 133-dimensional chemical atom descriptors for the atom GCN
- 3D coordinates used only for pairwise distance-based attention biases
- 37-dimensional bond-node descriptors
- 2,048-dimensional RDKit topological fingerprint
- species, sex, and species-sex interaction embeddings

Three-dimensional conformers are generated with RDKit. Embedding uses standard
distance geometry, fixed-seed random coordinates, and ETKDG as sequential
fallbacks, with at most 10 attempts or iterations per stage. Successfully
embedded conformers are optimized with MMFF94s for at most 100 iterations. If
force-field optimization fails, the embedded conformer is retained; if all
embedding stages fail, zero-valued 3D geometry features are used.

## Training

One invocation performs one run. The public command generates its random state
at runtime; the experimental random-state values used for the manuscript are
not included in this repository.

```bash
biocarcnet-train \
  --data-path data/biocarcnet.csv \
  --split-type record
```

Use `--split-type chemical` for chemical-level splitting. Training outputs are
written under `outputs/` and include the selected model and an exact split
manifest. The manifest uses the subset names `train`, `test`, and `validation`.
Final validation predictions and endpoint-specific RA, RBA, RP, RTP, RTN, F1,
AROC, and APR values are saved as CSV files.

The default objective combines label-smoothed focal binary cross-entropy,
`1.5` times the same-chemical cross-condition separation loss, and `1.0` times
an explicit positive-negative ranking loss. Use `--ranking-loss-weight 0` to
disable the ranking term.

## Prediction

Prediction requires a checkpoint produced by the training command:

```bash
biocarcnet-predict "CCO" \
  --checkpoint outputs/checkpoints/<run>/model.pt \
  --output predictions.csv
```

If species or sex is omitted for a single SMILES, predictions are generated for
all corresponding biological conditions. CSV, TSV, SMI, and plain-text inputs
are supported by the base installation; Excel input requires the `io` extra.

## Tests

```bash
pytest
```

The tests cover data validation, both splitting strategies, molecular feature
generation, the training objective, and a model forward pass.
