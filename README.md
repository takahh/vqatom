# VQ-Atom

VQ-Atom converts a molecule into one discrete token per atom. The current paper uses normalized bond-path descriptors (`bondpath_full_v1`, CB5k) and evaluates the tokens with a shared Transformer-based drug–target interaction (DTI) model.

## Required files

| Task | Required artifact | Public download |
|---|---|---|
| Tokenize a SMILES string | Corrected, normalized CB5k checkpoint | Pending |
| Reproduce a DTI experiment | Frozen bundle containing CSVs, model sources, and `manifest.json` | Pending |

The commands below require these artifacts. Checkpoint identity and bundle contents are documented in the [reproduction guide](dti/docs/reproduction.md#artifact-availability).

## 1. Set up token inference

Use Python 3.11 for the commands below. The CPU example was verified in a fresh Python 3.11.13 environment on macOS with four pinned packages.

```bash
git clone https://github.com/takahh/vqatom.git
cd vqatom
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

## 2. Tokenize one molecule

Place the corrected CB5k checkpoint at the path shown below, then run inference on methyl tert-butyl ether.

```bash
python infer_one_smiles_bondpath.py \
  --ckpt checkpoints/bondpath_full_v1/bondpath_full_v1_h2l2_cb5000.pt \
  --smiles 'COC(C)(C)C'
```

The JSON output contains `tokens` with the following IDs in input atom order.

```json
[3028, 4863, 2547, 3261, 3261, 3261]
```

Descriptor settings and normalization are restored from the checkpoint. Inference checks atom counts, token-ID ranges, and vocabulary consistency.

## 3. Run one DTI experiment

DTI uses a separate GPU environment with Python 3.11 and at least 40 GiB of VRAM. Follow the [GPU environment setup](dti/docs/reproduction.md#environment), then unpack the matching experiment bundle and set its path below.

```bash
BUNDLE=/path/to/bondpath_h2l2_cb5000_kiba_split1
python dti/tools/run_matched_v2.py \
  --bundle "$BUNDLE" \
  --dataset kiba --cold-type protein_cold --method vq_cb5000 \
  --split 1 --seed 0 --epochs 20 --out-root results
```

This verifies the bundle and prints the training command. To start training, repeat it with `--execute` appended. Results are written under `results/<task-name>/`.

## Read the results

Select the epoch with the highest `valid_metrics.auroc`, using the earliest epoch in a tie. From that same `epoch_*.json`, read `final_eval_metrics.auroc` and `final_eval_rank_metrics.adjusted_ndcg10`. The paper includes runs that finish their assigned budget and pairs methods by dataset, cold condition, split, and seed.

## Benchmark and further details

The main benchmark uses KIBA and Metz, protein-cold and scaffold-cold, splits 1–3, and seeds 0–2. Epoch budgets are 20 for Morgan and VQ-Atom, 50 for Continuous, and 10 for SMILES.

- [Reproduction guide](dti/docs/reproduction.md) covers GPU packages, model settings, other representations, codebook fitting, and artifact hashes.
- [Corpus provenance](dti/docs/corpus_provenance.md) documents the ChEMBL/PDBbind v2020 reference pool and verified combined counts.
- [Historical workflow](dti/docs/legacy_workflow.md) preserves the original GINE implementation and earlier experiments.
