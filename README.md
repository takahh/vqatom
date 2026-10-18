# VQ-Atom: Discretization and Drug–Target Interaction Benchmarking

VQ-Atom learns discrete, context-dependent atom tokens from molecular graphs. This guide starts with an overview of the architecture, followed by a practical end-to-end workflow, a file reference, and the detailed implementation and benchmark methodology.

## Contents

1. [Model overview](#1-model-overview)
2. [Installation and dependencies](#2-installation-and-dependencies)
3. [Quick start: end-to-end workflow](#3-quick-start-end-to-end-workflow)
4. [Repository and file guide](#4-repository-and-file-guide)
5. [Discretization: implementation details](#5-discretization-implementation-details)
6. [Dataset preparation: implementation details](#6-dataset-preparation-implementation-details)
7. [DTI model and benchmarking details](#7-dti-model-and-benchmarking-details)

---

## 1. Model overview

The workflow has three connected stages: **(1) train or load the VQ-Atom quantizer, (2) assign atom token IDs and prepare protein–ligand pairs, and (3) train and evaluate a DTI model**. The DTI benchmark also supports continuous atom features, SMILES, and atom-wise radius-2 Morgan tokens as alternative ligand representations.

```text
SMILES ──► molecular graph ──► three-layer GINE ──► atom vectors
              │                                      │
              └── intrinsic atom keys ──► codebooks ─┤
                                                     ▼
                                            local code indices
                                                     │
                                             codebook offsets
                                                     ▼
                                             global atom IDs
                                                     │
Protein sequence ──► ESM-2 ──────────────► DTI interaction model
                                                     │
                                                     ▼
                                            prediction / metrics
```

For the mathematical and architectural explanation, see [Section 5](#5-discretization-implementation-details) and [Section 7](#7-dti-model-and-benchmarking-details).
## Data Availability

The datasets used for the VQ-Atom DTI benchmarks are available on Hugging Face:

**[VQ-Atom DTI Dataset](https://huggingface.co/datasets/takahh/vqatom_data)**

The repository provides KIBA and Metz datasets with protein-cold, ligand-cold, and double-cold train/validation/test splits.

The same data splits are used across the compared ligand representation methods to ensure consistent evaluation.

### Download

```bash
pip install -U huggingface_hub

hf download takahh/vqatom_data \
  --repo-type dataset \
  --local-dir data
```

## 2. Installation and dependencies

> **Environment record (18 October 2026):** Package versions below were reported from an active GPU environment. This is a recorded environment, not a verified clean installation or a guarantee that all benchmark runs used identical versions. RDKit and Biopython were not present in the supplied `pip freeze` output; their versions remain unverified. MMseqs2 is a separate executable.

### System prerequisites

- Linux or macOS, Python 3, and a compatible PyTorch installation. A CUDA-capable GPU is strongly recommended for ESM-2 and DTI training.
- Git for cloning the repository.
- **MMseqs2** for the protein clustering/split-generation stage (not required merely to read prepared CSVs).
- Access to the matching VQ-Atom checkpoint and codebook/key-to-ID mapping for token inference.

### Clone and create an isolated environment

```bash
git clone https://github.com/takahh/vqatom.git
cd vqatom
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install Python packages

The reported environment uses **PyTorch 2.14.1+cu132**, **PyTorch Geometric 2.8.1**, **Transformers 5.20.1**, **Accelerate 1.15.0**, **NumPy 2.5.4**, **pandas 3.0.6**, and **scikit-learn 1.9.1**. The exact CUDA-specific PyTorch wheel may require a compatible package index; verify availability and follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/) rather than assuming a standard PyPI installation will resolve `+cu132`.

Once the matching PyTorch installation is available, install the other pinned packages:

```bash
python -m pip install -r requirements.txt
```

The requirements file pins directly relevant packages observed in the reported environment, but **does not constitute a complete lockfile**. Install RDKit and Biopython separately for preprocessing, after checking compatible versions, and record their versions. For PyTorch Geometric compatibility see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Non-Python tool for dataset generation

Install MMseqs2 using its [official installation instructions](https://github.com/soedinglab/MMseqs2) and verify it is on the path:

```bash
mmseqs --version
```

MMseqs2 is only needed for workflows that regenerate protein-clustered splits. Users of the published precomputed datasets may skip that step.

### Check the environment and record exact versions

```bash
python -m pip check
python -m pip freeze > installed_versions.txt
python - <<'PYTHON'
import sys
import torch
import torch_geometric
import transformers
import rdkit
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('PyTorch Geometric:', torch_geometric.__version__)
print('Transformers:', transformers.__version__)
print('RDKit:', rdkit.__version__)
PYTHON
```

For the paper's reproducibility package, confirm the versions on the machines used for the reported experiments and publish a validated environment lockfile. Record GPU model, CUDA/driver versions, and the relevant checkpoint hashes as well.

### External artifacts and current limitations

The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/takahh/vqatom_data) as described above. Some provided benchmark shell scripts still download inputs from a separately configured FTP server; they will not run unchanged on a clean checkout without replacing that step with local dataset paths. The dataset-generation script contains absolute local paths, and single-molecule inference has a documented interface mismatch. See the Quick start and the detailed notes below before attempting a complete reproduction.

---

## 3. Quick start: end-to-end workflow

> **Reproduction status:** These are source-derived command templates, **not a verified one-command installation**. The preparation script contains machine-specific paths; VQ tokenization requires a compatible checkpoint and `vqatom_module`; the single-SMILES utility has a documented return-value mismatch that must be fixed before running. The launchers also rely on external FTP storage. Adjust the environment and inputs first.

### Step 0 — Prerequisites

- Follow [Installation and dependencies](#2-installation-and-dependencies), including the system-specific PyTorch setup and the companion `requirements.txt`.
- Prepare the pretrained VQ-Atom checkpoint and its exact codebook/key mapping.
- Provide GraphDTA KIBA input files (`ligands_can.txt`, `proteins.txt`, `Y`) for the supplied data-generation script.
- Edit the absolute input/output paths in `make_train_data_etc.py`, and configure `vqatom_module.encode_smiles_to_atom_tokens` to use the same VQ-Atom model and vocabulary mapping.

### Step 1 — Discretize a single molecule and inspect its IDs

From the repository root, after fixing the inference interface described below:

```bash
python infer_one_smiles.py \
  --ckpt path/to/compatible_checkpoint.pt \
  --smiles 'CCO' \
  --device 0 \
  --hidden_dim <TRAINED_DIM> \
  --codebook_size <TRAINED_SETTING>
```

The quantizer assigns a **local index within a key-specific codebook**; adding the codebook offset produces the **global atom token ID** used downstream. `infer_one_smiles.py` currently expects three return values while `new_train_and_eval.evaluate()` returns four ID-related values in the supplied version; reconcile them before treating this invocation as executable.

### Step 2 — Generate ligand token IDs and split CSVs

The root-level script includes unique-SMILES extraction, VQ-Atom tokenization, graph annotations, MMseqs2 clustering, and protein-cold train/validation/test partitioning:

```bash
python make_train_data_etc.py
```

It currently uses constants rather than CLI flags. The generated `lig_tok` column contains VQ-Atom token IDs. For DTI training, place or rename the generated CSVs to match the expected structure:

```text
data/kiba/protein_cold/split_1/
  kiba_protein_cold_split_1_train.csv
  kiba_protein_cold_split_1_valid.csv
  kiba_protein_cold_split_1_test.csv
```

The supplied root script primarily creates **protein-cold** data. Ligand-cold and double-cold datasets require separately prepared split files and leakage checks. The source's split seed and the DTI model's training seed are independent.

### Step 3 — Train the DTI model using the generated IDs

Minimal local-input template, executed from the repository root:

```bash
cd dti
python dti.py \
  --ligand_mode vqatom \
  --train_csv ../data/kiba/protein_cold/split_1/kiba_protein_cold_split_1_train.csv \
  --valid_csv ../data/kiba/protein_cold/split_1/kiba_protein_cold_split_1_valid.csv \
  --final_eval_csv ../data/kiba/protein_cold/split_1/kiba_protein_cold_split_1_test.csv \
  --use_train_valid_csv \
  --vq_ckpt ../data/model_epoch_3.pt \
  --y_thr 12.1 \
  --select_on auroc \
  --epochs 20 \
  --seed 0 \
  --out_dir ./kiba_protein_cold_vqatom_split1_seed0
```

This is **not** the full experimental configuration. For matched benchmark settings, inspect the four `run_*.sh` launchers. Their defaults are VQ-Atom **20**, continuous **40**, SMILES **5**, and Morgan **40** epochs. The scripts expect external FTP credentials and dataset artifacts; do not commit credentials.

### Step 4 — Inspect metrics

From the repository root:

```bash
python dti/plot_metrics.py \
  --json_dir dti/kiba_protein_cold_vqatom_split1_seed0
```

This creates `auc_curve.png` and `loss_curve.png` from per-epoch JSON logs. Use **validation AUROC** for checkpoint selection, not test AUROC.

## 4. Repository and file guide

| Location | Purpose |
| --- | --- |
| `train_teacher.py` | VQ-Atom training entry point |
| `models.py` | Atom embedding and three-hop GINE encoder |
| `vq.py` | Vector quantization and codebook operations |
| `new_train_and_eval.py` | Discretization training/evaluation logic |
| `utils.py`, `make_cbdict.py` | Key-specific codebook sizes |
| `infer_one_smiles.py` | Single-molecule token inference |
| `make_train_data_etc.py` | Dataset creation; located at repository root |
| `dti/dti.py` | DTI training and evaluation |
| `dti/model.py` | Ligand encoder imported by DTI script |
| `dti/smiles_mlm.py` | SMILES tokenizer and MLM utilities |
| `dti/run_vqatom.sh` | VQ-Atom benchmark launcher |
| `dti/run_continuous.sh` | Continuous baseline launcher |
| `dti/run_smiles.sh` | SMILES baseline launcher |
| `dti/run_morgan.sh` | Morgan baseline launcher |
| `dti/overlap_check.py` | Protein-cluster, sequence and pair overlap checks |
| `dti/get_data_statistics_for_neurips.py` | Dataset statistics |
| `dti/plot_metrics.py` | Epoch-level AUROC and loss plots |
| `dti/summarize_results.py` | Aggregate run metrics; paths currently hard-coded |

The `run_*.sh` names are canonical names suggested for the uploaded timestamped scripts; verify filenames in your checkout. Some utilities still contain absolute local paths and require configuration before use.

---

## 5. Discretization: implementation details

VQ-Atom converts **context-dependent atom representations** into **discrete token IDs**. Unlike an atom-type lookup, each token reflects both an atom's intrinsic chemical attributes and its surrounding molecular graph.

This part describes the discretization pipeline in the accompanying implementation. Downstream drug–target interaction prediction is covered in Part III.

### Pipeline

```text
SMILES
  │
  ▼
Molecular graph (atoms, bonds, atom attributes)
  │
  ├── Intrinsic atom key ────────────────┐
  │   (Z, charge, hybridization,         │
  │    aromaticity, ring, H count)       │
  ▼                                      ▼
Atom feature embedding             Key-specific codebook
  │                                      ▲
  ▼                                      │
Three GINE message-passing layers        │
  │                                      │
  ▼                                      │
Combine 0-, 1-, 2-, 3-hop states         │
  │                                      │
  ▼                                      │
Project + layer norm + L2 norm           │
  │                                      │
  └──────────► Vector quantization  ◄────┘
                       │
                       ▼
            Local code index (integer)
                       │
                       ▼
             Global atom token ID
```

### From SMILES to a molecular graph

Each molecule is represented as a graph: atoms are nodes and chemical bonds are edges. The graph-construction pipeline supplies an adjacency matrix and an atom-feature matrix. The one-molecule inference utility expects **79 input attributes per atom** and supports padding to **100 atoms** by default. These are implementation defaults, not fundamental limits of vector quantization.

The feature matrix contains both intrinsic atomic information and descriptors of the local chemical environment. Bond types are also embedded for graph message passing.

### Intrinsic keys select codebooks

Before quantization, each atom is assigned a **key** based on six intrinsic attributes:

| Field | Meaning |
| --- | --- |
| `Z` | Atomic number |
| `charge` | Formal charge |
| `hyb` | Encoded hybridization state |
| `arom` | Aromaticity flag |
| `ring` | Ring-membership flag |
| `hnum` | Encoded hydrogen count |

The implementation represents a key as an underscore-separated string, such as `6_0_3_1_1_0`. The numeric values are the feature encodings used by the preprocessing pipeline; they should not all be interpreted as literal chemical quantities.

**Crucially, the key does not include the full neighboring environment.** Degree, ring size, functional-group indicators and other contextual features are instead supplied to the graph encoder. Thus atoms sharing a key can still receive different discrete IDs because their molecular environments differ.

### Graph encoding

The encoder (`EquivariantThreeHopGINE` in `models.py`) embeds the atom attributes, then applies **three GINE graph-convolution layers** with bond embeddings. It combines the initial node state and the outputs of all three layers, projects them to the quantization dimension, applies layer normalization, and L2-normalizes the resulting vectors.

For atom `i`, denote the resulting continuous vector by `z_i`.

### Vector quantization

Each intrinsic key `k` has its own collection of learned code vectors (a **codebook**):

$$
C_k = \{c_{k,0}, c_{k,1}, \ldots, c_{k,K_k-1}\}.
$$

For an atom with key `k`, vector quantization selects a code within that key's codebook and produces a **local integer code index** `j_i`:

$$
j_i = Q_k(z_i), \qquad j_i \in \{0,\ldots,K_k-1\}.
$$

Conceptually, this assigns a continuous representation to a learned representative vector. The codebook vectors are continuous parameters; **the discrete integer is the index of the selected entry**, not a numerical rounding of the atom embedding. The integer's magnitude has no inherent chemical meaning.

The implementation includes codebook initialization and training mechanisms in `vq.py`, including K-means-based initialization and quantization-related losses. Exact assignments depend on the trained checkpoint and quantizer configuration.

### Local indices versus global token IDs

Local code indices are only unique **within their codebook**. To obtain a token vocabulary shared across all atom keys, the inference utility assigns an offset to each codebook:

$$
\mathrm{token}_i = \mathrm{offset}(k_i) + j_i.
$$

Offsets are cumulative codebook sizes in the checkpoint's codebook iteration order. Therefore, a token ID must be interpreted together with the **same codebook configuration and ordering** used to generate it.

#### Illustrative example (not a measured model output)

Suppose a carbon atom has key `6_0_3_1_1_0`. Its graph representation is assigned local code index `17`. If this key's codebook begins at global offset `5000`, then its final token ID is `5017`.

```text
Atom key:         6_0_3_1_1_0
GNN vector:       [continuous learned features ...]
Selected code:    17         (local index)
Codebook offset:  5000       (illustrative)
Global token ID:  5017
```

Two atoms with the same key can select different local codes; atoms with different keys can have the same local index but different global IDs.

### What the tokens mean

- A token identifies a **learned quantization cell** for an atom within a particular intrinsic-attribute group.
- Its assignment can depend on the atom's **graph neighborhood**, not merely its elemental identity.
- Token numbers are **categorical identifiers**, not physical measurements or ordinal chemical descriptors.
- Token IDs are only directly comparable when generated with a **consistent checkpoint and vocabulary mapping**.

---

## 6. Dataset preparation: implementation details

This document describes the dataset-generation pipeline implemented in the repository-root script `make_train_data_etc.py`. The supplied script is primarily a **KIBA / GraphDTA to VQ-Atom DTI CSV** pipeline, ending with a protein-cold split. It contains optional structural annotation utilities; it is **not** a generic command-line generator for all datasets and cold-split types.

### Processing flow

```text
GraphDTA KIBA inputs
  |-- ligands_can.txt
  |-- proteins.txt
  `-- Y
          |
          v
  Protein–ligand interaction rows
          |
          v
  SMILES canonicalization and unique-ligand collection
          |
          v
  VQ-Atom tokenization of unique SMILES
          |
          v
  Join token IDs, graph and optional annotations
          |
          v
  Protein sequence clustering (MMseqs2, 40% identity setting)
          |
          v
  Protein-cold train / validation / test CSVs
```

The script's `main()` calls `pass1_from_graphdta_processed()`, `pass2_make_smiles_token_map_parallel()`, `pass3_join_tokens()`, and `stage4_protein_cold_split()` in that order.

### Inputs and dependencies

The default paths are **machine-specific constants** near the beginning of `make_train_data_etc.py`:

```python
GRAPHDTA_DIR = Path("/Users/taka/Documents/GraphDTA/data/kiba")
OUT_DIR = Path("/Users/taka/Desktop/kiba_fast_out")
MMCIF_DIR = Path("/Users/taka/Documents/mmcif")
```

Change these before running. Expected GraphDTA KIBA inputs include `ligands_can.txt`, `proteins.txt`, and `Y`. The script imports RDKit, Biopython, pandas-related data processing utilities, and the local `vqatom_module.encode_smiles_to_atom_tokens` interface. MMseqs2 is used for protein clustering; optional structure-related logic reads mmCIF/PDB information.

**Important:** `vqatom_module` and the matching VQ-Atom checkpoint must be available and configured in the user's environment. The supplied file alone does not make token generation fully standalone.

### Configuration

Selected settings from the supplied script:

| Setting | Value | Meaning |
| --- | --- | --- |
| `Y_THR` | `12.1` | Binary classification threshold for the prepared KIBA scores |
| `SPLIT_RATIOS` | `0.8 / 0.1 / 0.1` | Target train/valid/test proportions |
| `SPLIT_SEED` | `123` | Dataset split seed |
| `MMSEQS_MIN_SEQ_ID` | `0.40` | MMseqs2 clustering identity parameter |
| `MMSEQS_C` | `0.8` | MMseqs2 coverage parameter |
| `MIN_SEQ_LEN` | `50` | Minimum sequence length |
| `MAX_SEQ_LEN` | `1022` | Maximum sequence length |
| `CONTACT_CUTOFF` | `4.5` | Distance cutoff used in contact annotation code |
| `USE_GENERIC_MURCKO` | `False` | Scaffold representation choice |

The script includes additional component pruning, filtering, and balancing logic. The 80/10/10 values are **targets**: cluster/group constraints can prevent exact row-level proportions.

### Molecular and protein preprocessing

**Ligands:** SMILES are parsed and canonicalized using RDKit, and unique SMILES are tokenized via `encode_smiles_to_atom_tokens`. Molecular graph edges are exported as directed atom-pair indices. Atom features are also serialized for downstream graph-based ligand modes.

**Proteins:** sequences are cleaned/validated, written to FASTA, and clustered using MMseqs2. The script uses the resulting sequence-to-cluster mapping to construct a protein-cold partition.

**Optional structure information:** the script contains mmCIF/PDB parsing and protein–ligand contact logic. Contact annotations depend on the availability and suitability of the corresponding structures; do not assume every row has a valid structure or contact mask.

### Intermediate outputs

The configured output directory contains intermediate products such as:

```text
pass1_rows.csv
unique_smiles.csv
smiles_tok_map.csv
kiba_tok.csv
kiba_tok_annotated.csv
protein_unique_seqs.fasta
seq_to_cluster_40id.csv
split_components_summary.csv
train.csv
valid.csv
test.csv
```

The script also creates MMseqs2 intermediate directories and other caches as needed. These files allow intermediate stages to be inspected independently.

### Final CSV format

The final split writer emits these columns:

| Column | Meaning |
| --- | --- |
| `seq` | Protein amino-acid sequence |
| `smiles` | Ligand SMILES |
| `lig_tok` | VQ-Atom atom-token IDs |
| `edge_index` | Serialized molecular graph edges |
| `atom_feat` | Serialized atom features |
| `aff_type` | Affinity measurement type, if available |
| `aff_nm` | Affinity value in nM, if available |
| `y` | Numeric model target |
| `src_pdbid` | Associated structure ID, if available |
| `src_chain` | Associated protein chain, if available |
| `contact_mask` | Optional residue contact annotation |
| `contact_n` | Contact count / summary |

The downstream DTI scripts additionally support `morgan_atom_r2_token` and a SMILES vocabulary. **These are not among the final CSV columns emitted by this script's `write_split_csvs()` function**; prepare them with the corresponding preprocessing utilities before Morgan or SMILES experiments.

### Leakage checks and alternate splits

The supplied `dti/overlap_check.py` checks protein-cluster, exact protein sequence, and exact protein–ligand pair overlap among train/valid/test sets. Run an appropriate overlap audit after creating each split.

**Protein-cold:** the supplied `make_train_data_etc.py` implements a clustering-based protein-cold pipeline. The actual partition must be audited rather than assumed leakage-free solely from its name.

**Ligand-cold and double-cold:** the DTI launchers accept these split names, but the supplied data-generation script does not establish their complete construction. Document and distribute the corresponding split generators separately. For ligand-cold, ensure ligand entities are separated; for double-cold, verify both protein and ligand separation according to the declared entity or cluster definitions.

### Reproducibility checklist

- Record the GraphDTA source/version and input checksums.
- Record RDKit, Biopython, MMseqs2, and Python versions.
- Preserve the VQ-Atom checkpoint and token-ID mapping used for `lig_tok`.
- Preserve split seed, protein clustering parameters, and filtering settings.
- Check overlap of sequences, clusters, ligands, and exact interaction pairs as appropriate.
- Save dataset statistics and split sizes after filtering.
- Publish the exact generated CSVs or a complete, executable generation recipe when licensing permits.

See [Part III: DTI benchmarking](#part-iii-dti-benchmarking) for training and evaluation.

---

---

## 7. DTI model and benchmarking details

This document describes the DTI experiments implemented in `dti/`, including ligand-representation comparisons, protein encoding, training, and evaluation. The experiments compare **VQ-Atom**, **continuous atom representations**, **SMILES**, and **atom-wise radius-2 Morgan tokens** under a shared DTI training framework.

> **Scope:** This is an implementation guide, not a claim that all experimental results or checkpoints are included in the repository. The commands below assume the run scripts are placed in `dti/` and that the required data and checkpoints have been prepared.

### Overview

```text
Ligand (SMILES)                       Protein (amino-acid sequence)
       |                                          |
       +-- VQ-Atom / continuous / SMILES / Morgan  +-- ESM-2
       |                                          |
       v                                          v
 Ligand representation                     Protein representation
       |                                          |
       +--------------- DTI interaction model ----+
                              |
                              v
                  Interaction score / logit
                              |
                              v
                 Classification and ranking metrics
```

The main current training script is `dti/dti.py` (corresponding to the supplied `dti(8).py`). It supports ligand modes `vqatom`, `continuous`, `smiles`, and `morgan`, alongside additional pretrained variants. The default protein backbone in the code is `facebook/esm2_t33_650M_UR50D`.

For the current classification experiments, labels are defined by `y_bin = 1` when `y >= y_thr` and `0` otherwise. **The supplied run scripts default to `Y_THR=12.1` for both KIBA and Metz**; confirm that the target scale and threshold are appropriate for each dataset before running.

### Data and input format

The training commands expect separate train, validation, and test CSV files. Relevant columns depend on the ligand mode:

| Column | Purpose |
| --- | --- |
| `seq` | Protein amino-acid sequence |
| `smiles` | Ligand SMILES string |
| `y` | Numeric interaction score; binarized using `--y_thr` |
| `lig_tok` | Precomputed VQ-Atom token IDs |
| `edge_index` | Molecular graph connectivity for graph-based modes |
| `atom_feat` | Atom-level features for graph-based modes |
| `morgan_atom_r2_token` | Atom-wise Morgan radius-2 dense IDs for Morgan mode |

Other optional fields can carry structural/contact annotations. See `dti/dti.py` for the precise mode-specific requirements.

The launchers expect the following layout:

```text
data/
└── kiba/                         # or metz/
    └── protein_cold/             # or ligand_cold/, double_cold/
        └── split_1/
            ├── kiba_protein_cold_split_1_train.csv
            ├── kiba_protein_cold_split_1_valid.csv
            └── kiba_protein_cold_split_1_test.csv
```

Each dataset/split combination must have matching CSVs and preprocessing artifacts. The current launchers fetch data from a separately configured FTP server; **those files are not automatically provided by cloning this repository**. For public reproduction, replace the private data-transfer step with locally prepared files or an accessible dataset distribution.

### Compared ligand representations

| Mode | `--ligand_mode` | Additional input |
| --- | --- | --- |
| VQ-Atom | `vqatom` | Precomputed `lig_tok` and compatible VQ checkpoint metadata |
| Continuous | `continuous` | Molecular graph/atom features; continuous representation setup |
| SMILES | `smiles` | SMILES strings and `smiles_vocab.json` |
| Morgan | `morgan` | `morgan_atom_r2_token` and matching Morgan vocabulary CSV |

VQ-Atom uses discrete atom identifiers learned from molecular contexts. Continuous mode provides a non-quantized graph-based comparison. SMILES and Morgan provide alternative ligand tokenizations. The representation mode is changed while the DTI training configuration is kept largely consistent; **epoch budgets are not identical** in the supplied launchers.

The SMILES tokenizer and masked-language-model utilities are in `dti/smiles_mlm.py`. Its vocabulary must match the tokenizer used by the DTI run. The Morgan launcher validates that its vocabulary uses dense mapped IDs `1..N` and that all three CSVs contain valid IDs.

### Split strategies and evaluation

The experiment interface recognizes:

- **Protein-cold:** hold out protein sequences or protein clusters across partitions.
- **Ligand-cold:** hold out ligands across partitions.
- **Double-cold:** hold out both protein and ligand entities across partitions.

The exact construction and leakage criteria must be documented alongside each published split. The supplied root-level `make_train_data_etc.py` implements a **protein-cold preparation workflow**; it does **not** by itself establish how the ligand-cold and double-cold CSVs were generated.

`dti.py` supports model selection using validation AUROC, AP, or F1 (`--select_on`). The supplied launchers select **validation AUROC**. Per-epoch JSON results expose validation and final-evaluation metrics, including AUROC and enrichment/ranking-related metrics. When reporting test results, select the checkpoint using validation data, not test performance.

### Audit utilities

`overlap_check.py` inspects 40%-identity protein-cluster overlap, exact sequence overlap, and exact `(seq, smiles)` pair overlap between partitions. Its input paths are currently hard-coded and must be edited for a new dataset.

`get_data_statistics_for_neurips.py` summarizes sample counts, unique proteins/ligands, positive rates, and per-entity distributions. It also currently uses local hard-coded paths and `Y_THR=12.1`.

### Inspecting results

The launcher outputs are named like:

```text
dti/kiba_protein_cold_vqatom_split1_seed0/
    epoch_001.json
    epoch_002.json
    ...
```

To plot AUROC and loss curves for a run:

```bash
python dti/plot_metrics.py \
  --json_dir dti/kiba_protein_cold_vqatom_split1_seed0
```

This produces `auc_curve.png` and `loss_curve.png` in the specified run directory. The plot labels distinguish training, validation, and final-evaluation AUROC.

`dti/summarize_results.py` reads `epoch_*.json` from a configured set of run directories and summarizes metrics including AUROC, enrichment factors, and ranking metrics. **Its paths and run names are currently hard-coded**; update them before using it with newly generated runs.

### Reproducibility notes

- Keep dataset versions, split definitions, seeds, label thresholds, and model checkpoints fixed when comparing ligand representations.
- Reuse the exact SMILES/Morgan vocabulary mapping associated with each experiment; token IDs are not portable between unrelated vocabularies.
- For VQ-Atom, use the same discretization checkpoint and global ID mapping as were used to prepare `lig_tok`.
- Preserve validation-based checkpoint selection and report variation across seeds and splits.
- Record software versions, model initialization, GPU hardware, and all training flags for published comparisons.
- Avoid relying on private FTP paths or local absolute paths for a public reproduction package.
