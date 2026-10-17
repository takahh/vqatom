# VQ-Atom: Molecular Atom Discretization

VQ-Atom converts **context-dependent atom representations** into **discrete token IDs**. Unlike an atom-type lookup, each token reflects both an atom's intrinsic chemical attributes and its surrounding molecular graph.

This README describes the discretization pipeline in the accompanying implementation. It does not describe downstream drug–target interaction prediction.

## Pipeline

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

## 1. From SMILES to a molecular graph

Each molecule is represented as a graph: atoms are nodes and chemical bonds are edges. The graph-construction pipeline supplies an adjacency matrix and an atom-feature matrix. The one-molecule inference utility expects **79 input attributes per atom** and supports padding to **100 atoms** by default. These are implementation defaults, not fundamental limits of vector quantization.

The feature matrix contains both intrinsic atomic information and descriptors of the local chemical environment. Bond types are also embedded for graph message passing.

## 2. Intrinsic keys select codebooks

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

## 3. Graph encoding

The encoder (`EquivariantThreeHopGINE` in `models.py`) embeds the atom attributes, then applies **three GINE graph-convolution layers** with bond embeddings. It combines the initial node state and the outputs of all three layers, projects them to the quantization dimension, applies layer normalization, and L2-normalizes the resulting vectors.

For atom `i`, denote the resulting continuous vector by `z_i`.

## 4. Vector quantization

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

## 5. Local indices versus global token IDs

Local code indices are only unique **within their codebook**. To obtain a token vocabulary shared across all atom keys, the inference utility assigns an offset to each codebook:

$$
\operatorname{token}_i = \operatorname{offset}(k_i) + j_i.
$$

Offsets are cumulative codebook sizes in the checkpoint's codebook iteration order. Therefore, a token ID must be interpreted together with the **same codebook configuration and ordering** used to generate it.

### Illustrative example (not a measured model output)

Suppose a carbon atom has key `6_0_3_1_1_0`. Its graph representation is assigned local code index `17`. If this key's codebook begins at global offset `5000`, then its final token ID is `5017`.

```text
Atom key:         6_0_3_1_1_0
GNN vector:       [continuous learned features ...]
Selected code:    17         (local index)
Codebook offset:  5000       (illustrative)
Global token ID:  5017
```

Two atoms with the same key can select different local codes; atoms with different keys can have the same local index but different global IDs.

## 6. Training and inference

The training entry point is `train_teacher.py`, which constructs `EquivariantThreeHopGINE` and invokes the training workflow in `new_train_and_eval.py`. The codebook-size mapping `CBDICT` is defined in `utils.py`; `make_cbdict.py` provides a helper to propose per-key sizes from observed frequencies. Quantization logic is implemented in `vq.py`.

For a single molecule, the supplied inference utility is `infer_one_smiles.py`. After preparing a **compatible checkpoint** and the graph preprocessing module (`smiles_to_npy.py`), the command has the following form:

```bash
python infer_one_smiles.py \
  --ckpt path/to/compatible_checkpoint.pt \
  --smiles 'CCO' \
  --device 0 \
  --hidden_dim <TRAINED_DIM> \
  --codebook_size <TRAINED_SETTING>
```

Replace the bracketed placeholders with the values used during training. The script prints the SMILES, global token IDs, key IDs, local code IDs, and the size of the key mapping. The checkpoint architecture, feature layout, and codebook settings must match the training run.

**Implementation note:** The supplied files contain different inference return conventions: `new_train_and_eval.evaluate()` currently returns four ID-related values, while `infer_one_smiles.py` unpacks a legacy three-value form. Reconcile these interfaces before treating the example command as an end-to-end verified invocation.

## 7. What the tokens mean

- A token identifies a **learned quantization cell** for an atom within a particular intrinsic-attribute group.
- Its assignment can depend on the atom's **graph neighborhood**, not merely its elemental identity.
- Token numbers are **categorical identifiers**, not physical measurements or ordinal chemical descriptors.
- Token IDs are only directly comparable when generated with a **consistent checkpoint and vocabulary mapping**.

## Relevant files

| File | Role |
| --- | --- |
| `models.py` | Atom embeddings, three-hop GINE encoder, quantizer integration |
| `vq.py` | Codebooks, initialization, vector quantization, associated losses |
| `new_train_and_eval.py` | Graph conversion, intrinsic-key grouping, training/evaluation |
| `train_teacher.py` | Training entry point |
| `utils.py` | Codebook-size dictionary (`CBDICT`) |
| `make_cbdict.py` | Frequency-based codebook-size proposal utility |
| `infer_one_smiles.py` | Single-SMILES inference and global ID conversion |
| `args.py` | Command-line training parameters |

