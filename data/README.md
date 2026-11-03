# Modeling dataset

This directory contains the curated modeling dataset and the independent
ISSCAN validation dataset distributed with BioCarcNet.

## Modeling dataset

`biocarcnet.csv` contains 5,847 chemical-species-sex records for 2,314 unique
chemicals.

### Schema

```text
smiles,label,species,sex
```

- `smiles`: standardized canonical SMILES
- `label`: binary carcinogenicity label (`0=negative`, `1=positive`)
- `species`: rodent species (`0=rat`, `1=mouse`)
- `sex`: biological sex (`0=male`, `1=female`)

Records were grouped by canonical SMILES, species, and sex. Consistent
duplicates were merged. For conflicting outcomes within the same group, a
positive label was retained when any source record was positive; otherwise the
group was labeled negative.

### Integrity

- Rows: 5,847
- Unique canonical SMILES: 2,314
- SHA-256: `9bb216394c0272c516581020f14e9df00bcbaf1beb7869191c8cd8879cf6b168`

## Independent ISSCAN dataset

`external_isscan.csv` contains 119 experimental records for 52 unique chemicals
after removal of chemicals overlapping the modeling dataset. Its columns are:

```text
CAS,ChemName,smiles,label,species,sex
```

- Rows: 119
- Unique SMILES: 52
- SHA-256: `850ecc229a08511d7a1f742fa14430845eb54e7873f4417ccbbd8c7b8b0c9c6e`

## Generated split manifests

The training command generates a split manifest for each run. In accordance
with the manuscript, subset names are `train`, `test`, and `validation`; the
test set is used for model selection and the validation set for final
evaluation.

