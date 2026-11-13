# Detailed reproduction guide

VQ-Atom assigns one reusable token to each atom by clustering chemical-environment descriptors. The current paper evaluates a hydrogen-count-corrected, L2-normalized `bondpath_full_v1` tokenizer under a shared downstream DTI architecture.

Run all commands from the repository root. For the short execution sequence, see the [README](../../README.md).

## Current representation and benchmark

Each atom has 303 base descriptor coordinates. Bond-conditioned one-edge and two-edge aggregates receive weights 0.4 and 0.3. Joint path counts retain the ordered bond types and intermediate/terminal elements. Concatenation gives 232,939 sparse coordinates. The normalized CB5k centroids store the 1,587 coordinates active in the reference data. Each atom selects a nearest centroid within its intrinsic-property group.

The nominal CB5k checkpoint contains 5,021 atom IDs. Its 256-dimensional DTI embedding has 5,024 rows including special tokens. Morgan uses atom-wise radius-two identifiers. Continuous uses the same normalized descriptor vectors before token assignment. SMILES uses a separate token vocabulary and positional embeddings.

The main benchmark covers KIBA and Metz, protein-cold and scaffold-cold, splits 1–3, and seeds 0–2. The protein encoder, ligand Transformer body, and interaction modules are shared across representations. Epoch budgets are 10 for SMILES, 50 for Continuous, and 20 for Morgan and VQ-Atom. The highest validation-AUROC checkpoint selects the reported test metrics.

The shared ligand Transformer has six layers, width 256, eight heads, and internal dropout 0.1. The downstream interaction block contains two directional cross-attention modules, with eight heads per direction and dropout 0.3. Protein, ligand, and interaction learning rates are 5e-6, 1e-5, and 1e-4. The protein encoder is ESM-2 t33 650M, with its bottom 28 layers frozen. The batch size is 8, AdamW weight decay is 0.05, and linear warmup occupies 5% of optimizer steps. Classification, regression, and ranking loss weights are 0.05, 1.0, and 0.05. Contact-loss weight is zero.

## Artifact availability

| Artifact | Availability and identity |
|---|---|
| Current tokenizer implementation | `infer_one_smiles_bondpath.py` and `quantize/bondpath_*.py` |
| Current matched DTI runner | `dti/tools/run_matched_v2.py` and its hash-verified experiment bundle |
| Corrected normalized CB5k checkpoint | Retained experiment artifact `bondpath_full_v1_h2l2_cb5000.pt`, SHA256 `1ef7017740f12d6b7855d8d91c9a2c7ae544ea6135f618a9d20485b6eb078fee`. Public distribution remains to be completed. |
| Exact current DTI CSVs and frozen bundles | Retained with experiment manifests and file hashes. Public distribution remains to be completed. |
| Reference array-file manifest and initial group layout | `dti/docs/bondpath_cb5000_adj_attr_manifest.json` and `dti/docs/bondpath_cb5000_legacy_layout.json` |
| Reference molecule-list and shard audit | Procedure and measured counts in `dti/docs/corpus_provenance.md` |
| Earlier published DTI data | KIBA/Metz protein-cold, ligand-cold, and double-cold CSVs. These are the earlier release. The current scaffold-cold files and normalized-tokenizer artifacts require the current frozen bundles. |

The earlier public data repository is available on Hugging Face. [Published DTI data](https://huggingface.co/datasets/takahh/vqatom_data)

## Environment

The CPU tokenization example uses the smaller `requirements-inference.txt` file in the repository root. The environment below records the GPU DTI experiments.

The retained matched-run log records Linux, Python 3.11, and the container image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`. The directly used Python package versions from that run are pinned in `dti/docs/requirements_matched_20261113.txt`. Some packages were installed without a version constraint when the experiments ran. The pinned file records one observed environment and supports a reproducible installation target.

Inside that container, install the recorded packages and record the resulting environment.

```bash
python -m pip install -r dti/docs/requirements_matched_20261113.txt
python -m pip check
python -m pip freeze > installed_versions.txt
```

GPU experiments used at least 40 GiB of VRAM. Token inference can run on CPU. MMseqs2 is required when regenerating protein clusters. Prepared CSVs already contain their assigned partitions.

## Execution order

### 1. Obtain the matching artifacts

Place the trusted corrected CB5k checkpoint at `checkpoints/bondpath_full_v1/bondpath_full_v1_h2l2_cb5000.pt`. Obtain the exact prepared bundle for the dataset and split being evaluated. Each bundle contains `manifest.json`, the training sources, vocabularies or checkpoint metadata, and its CSVs. The manifest records every required file hash and task definition.

The earlier public dataset can be downloaded separately with the Hugging Face CLI.

```bash
hf download takahh/vqatom_data --repo-type dataset --local-dir data/earlier_release
```

Use the current experiment manifest to identify the CSVs and tokenizer identity for the expanded main benchmark.

### 2. Check SMILES token inference

```bash
python infer_one_smiles_bondpath.py \
  --ckpt checkpoints/bondpath_full_v1/bondpath_full_v1_h2l2_cb5000.pt \
  --smiles 'COC(C)(C)C'
```

The reference checkpoint returns `[3028, 4863, 2547, 3261, 3261, 3261]` in parsed SMILES atom order for methyl tert-butyl ether. Inference restores the descriptor settings and hydrogen-count version from the checkpoint. It verifies the atom/ID count, ID range, centroid shapes, and total vocabulary size.

### 3. Verify one matched DTI task

Set `BUNDLE` to the directory of the prepared KIBA split-1 CB5k package. This command verifies the frozen sources, CSVs, vocabulary metadata, and epoch policy, then prints the training command.

```bash
BUNDLE=artifacts/bondpath_h2l2_cb5000_kiba_split1
python dti/tools/run_matched_v2.py \
  --bundle "$BUNDLE" \
  --dataset kiba --cold-type protein_cold --method vq_cb5000 \
  --split 1 --seed 0 --epochs 20 --out-root results
```

Add `--execute` to start training after verification. Use an empty output directory. The runner rejects mismatched hashes and inconsistent task/epoch settings. Morgan, SMILES, and Continuous tasks use their corresponding prepared bundles and method names `morgan`, `smiles`, and `continuous`.

### 4. Collect the selected checkpoint metrics

Each task writes `epoch_*.json` files. In the expanded benchmark, select the epoch with maximum `valid_metrics.auroc` within the method's assigned budget, taking the earliest epoch in a tie. Read test AUROC from `final_eval_metrics.auroc` and adjusted NDCG@10 from `final_eval_rank_metrics.adjusted_ndcg10` at that same epoch. Main tables include runs with complete training histories. Paired comparisons require the same dataset, cold condition, split, and seed for both methods. Seeds within a split share test examples.

## Refit the codebook

The original reference arrays and their `smiles_*.txt` sidecars are required. The fitting script checks the adjacency and attribute hashes against the recorded manifest, repairs total hydrogen counts from the original SMILES, and verifies atom-count and atomic-number alignment.

```bash
python quantize/train_bondpath_codebook.py \
  --data artifacts/reference_shards \
  --layout dti/docs/bondpath_cb5000_legacy_layout.json \
  --reference-manifest dti/docs/bondpath_cb5000_adj_attr_manifest.json \
  --out checkpoints/reproduced_cb5000.pt \
  --device cuda --codebook-size 5000 --shards 192 --iters 20 --seed 0 \
  --correct-hydrogens --rebuild-layout --normalization l2
```

The selected corpus has 191,524 molecule records and 5,418,504 atom records. Source datasets are ChEMBL and PDBbind v2020. Their separate contributions are unavailable in the retained molecule list. The audit command reconstructs the recorded SMILES hashes from the retained combined input list.

```bash
python dti/tools/audit_bondpath_corpus.py \
  --smiles artifacts/discretize.noleak.final.smi \
  --checkpoint checkpoints/bondpath_full_v1/bondpath_full_v1_h2l2_cb5000.pt \
  --output results/corpus_audit.json
```

The audit rejects a shard hash or fitted-atom-count mismatch. Detailed reconstruction assumptions and the retained hashes are documented in the provenance report. [Corpus provenance](corpus_provenance.md)

## Verification record

On 14 November 2026, a fresh macOS Python 3.11.13 environment installed `requirements-inference.txt`, passed `pip check`, and reproduced all six MTBE token IDs. The same inference output had been verified on 13 November. The matched runner verified a retained KIBA protein-cold split-1/seed-0 bundle and printed its 20-epoch command. The corpus reconstruction matched all 192 stored SMILES SHA256 values and the fitted atom count. A fresh GPU installation and full benchmark rerun remain to be validated independently.
