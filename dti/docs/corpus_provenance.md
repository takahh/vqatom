# Reference corpus provenance

Audited on 13 November 2026 for the corrected, normalized `bondpath_full_v1` CB5k checkpoint.

## Source attribution

The author confirmed that the combined reference corpus was assembled from ChEMBL and PDBbind v2020, consistent with the earlier research discussion. Historical preparation records reference `smiles_chembl.csv`, `smiles_pdbbind.csv`, and an additional `chembl_1M_id_smiles.csv` input. The separately retained older extraction script points to a ChEMBL 34 SDF. The exact ChEMBL release of every record in the final combined list cannot be established from the retained list alone.

Source-specific contributions are unavailable because the final retained SMILES list has no source labels. The combined totals below were reconstructed from the checkpoint hashes. They do not establish separate ChEMBL and PDBbind counts. The total number of complexes in the original PDBbind release is a different quantity from the number of ligand records used here.

## Verified reconstruction

| Stage | Measured records |
|---|---:|
| Stored combined input SMILES | 208,812 |
| Distinct input SMILES strings | 208,812 |
| Size-filtered combined list | 208,524 |
| Excluded by atom count of at least 100 | 288 |
| Selected codebook-fitting molecule records | 191,524 |
| Distinct selected SMILES strings | 191,524 |
| Selected atom records | 5,418,504 |
| Selected SMILES shards matching checkpoint hashes | 192 / 192 |
| Permissive-parser records | 1 |

The retained input file is `discretize.noleak.final.smi`. The reconstruction shuffles its lines using Python's `random.Random(7)`, applies the historical fewer-than-100-atom filter, and groups the retained strings into batches of 1,000. One historical record requires the checkpoint's permissive parsing fallback with the current RDKit parser. Its retention is necessary to reproduce all stored SMILES hashes and the fitted atom count.

The fitting script sorts adjacency filenames lexicographically and selects the first 192. Thus `adj_10.npy` precedes `adj_2.npy`. The selected set includes `adj_208.npy`, whose SMILES sidecar has 524 records. This explains the total of 191,524 records across 192 shards. The comparison matched every reconstructed SMILES SHA256 to `reference_sources` in the checkpoint, and the atom total matched the sum of the per-group fitting statistics.

SMILES-string uniqueness is the measured deduplication property. The filename's `noleak` label alone does not establish the overlap criterion or molecular-standardization procedure used upstream. The retained records establish the selected inputs, parser behavior, and fitting counts. Detailed upstream filtering history and per-source counts remain incomplete.

## Identifiers

- Combined input SHA256 `34f3d11623034027663f82fa9548ef2f7ab11a82b88a072b7bf4997793d1a246`
- Corrected normalized CB5k checkpoint SHA256 `1ef7017740f12d6b7855d8d91c9a2c7ae544ea6135f618a9d20485b6eb078fee`

## Reproduce the audit

```bash
python dti/tools/audit_bondpath_corpus.py \
  --smiles artifacts/discretize.noleak.final.smi \
  --checkpoint checkpoints/bondpath_full_v1/bondpath_full_v1_h2l2_cb5000.pt \
  --output results/corpus_audit.json
```

The script checks every recorded SMILES shard hash and the fitted atom count. The original checkpoint also retains adjacency and attribute array hashes. The reconstruction audit checks the SMILES records and atom totals. Refit-time array verification uses `bondpath_cb5000_adj_attr_manifest.json`.

## Evidence retained

- Checkpoint `reference_sources`, `fit_stats`, and parser configuration.
- Local combined input list and its SHA256.
- Author confirmation of ChEMBL and PDBbind v2020 attribution on 13 November 2026.
- Earlier preparation code referencing the source SMILES files.
- Executed reconstruction report, with 192 checked shard digests.
