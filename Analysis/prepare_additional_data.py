import numpy as np
from rdkit import Chem
from matplotlib import pyplot as plt
from rdkit.Chem import rdmolops
import random
from collections import Counter
from scipy.sparse import csr_matrix
from icecream import ic as ice
import numpy as np
np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import HybridizationType
# Example atom-level class labels (one class per atom, using atomic number as class)
from rdkit import Chem
from rdkit.Chem import Draw

smiles_list = [
    # Esters
    "CC(=O)Oc1ccccc1C(=O)O",
    "COC(=O)c1ccccc1O",
    "CCOC(=O)c1ccccc1O",
    "CC(=O)Oc1ccccc1",
    "CC(C)OC(=O)c1ccccc1O",
    "CC(=O)Oc1ccc2c(c1)C=CC=C2",
    "CCCCOC(=O)c1ccccc1O",
    "COC(=O)c1ccccc1OC(=O)C",
    "COc1cc(C(=O)OC)ccc1OC",
     "CC(=O)OCc1ccccc1",

    # Amino
    "CC(=O)Nc1ccc(O)cc1",
    "Nc1ccc(O)cc1",
    "CC(=O)Nc1ccccc1N",
    "CC(=O)Nc1ccc(NC)cc1",
    "CC(=O)Nc1ccc(NC(C)=O)cc1",
    "Cc1ccc(N)cc1",
    "CC(=O)Nc1cc(N)ccc1",
    "Nc1ccc(N)cc1",
    "CC(=O)Nc1ccccc1",
    "Cc1ccc(N)cc1N",

    # Indole
    "CN(C)C(=O)c1ccc2[nH]ccc2c1",
    "c1cc2ccccc2[nH]1",
    "c1cc2c(cc1)cc[nH]2",
    "c1ccc2c(c1)[nH]cc2C",
    "COc1ccc2[nH]ccc2c1",
    "c1ccc2cccnc2c1",
    "c1ccc2[nH]c(C)cc2c1",
    "c1ccc2[nH]ccc2c1O",
    "c1cc2[nH]c(C)c(C)cc2c1",
    "c1ccc2[nH]ccc2c1CO",
    "c1ccc2[nH]c(C(=O)O)cc2c1",

    # Pyridine
    "C1=CC=NC=C1",
    "c1ccncc1",
    "c1ccncc1C",
    "c1cc(C)cnc1",
    "c1cnccc1Cl",
    "c1ccncc1O",
    "c1cc(CN)cnc1",
    "c1ccncc1C(=O)O",
    "c1ccncc1C(=O)NC",
    "c1cnccc1N",
    "c1ccncc1C(=O)OC",

    # Phenyl
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccccc1",
    "Cc1ccccc1",
    "COc1ccccc1",
    "CC(=O)c1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "c1ccc(cc1)C#N",
    "CC(=O)Oc1ccccc1",
    "c1ccc(cc1)N(=O)=O",

    # Carbonyl
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)OC",
    "CC(=O)C",
    "CC(=O)O",
    "CC(=O)N",
    "CC(=O)c1ccccc1",
    "O=CC1=CC=CC=C1",
    "O=C1CCCCC1",
    "CC(=O)OC(C)C",
    "CC(=O)N(C)C",
    "O=Cc1ccc(O)cc1",

    # Hydroxyl
    "CC(C)Cc1ccc(O)cc1O",
    "Oc1ccccc1",
    "CC(C)C(O)c1ccccc1",
    "COc1ccccc1O",
    "OC(C)c1ccccc1",
    "CC(O)c1ccc(O)cc1",
    "CC(O)C(C)(C)O",
    "CC(C)(O)c1ccccc1",
    "CC(O)CO",
    "OCC(O)CO",
    "CC(O)c1ccc(O)cc1C"
]

random.seed(7)

# Function to get SMILES from file
def get_smiles(ipath):
    with open(ipath) as f:
        smiles = [line.split(",")[1].strip() for line in f.readlines()]
    return smiles


def assign_node_classes(atom):
    # Use atomic number as the class (you can modify this logic based on your requirements)
    return atom.GetAtomicNum()


# Function to generate atom features as vectors
def atom_to_feature_vector(atom):
    # Example feature vector: atomic number, degree, formal charge, hybridization, is_aromatic, is_in_ring
    return np.array([
        atom.GetAtomicNum(),               # Atomic number
        atom.GetDegree(),                  # Degree (number of bonds)
        atom.GetFormalCharge(),            # Formal charge
        atom.GetHybridization().real,      # Hybridization state (numeric)
        int(atom.GetIsAromatic()),         # Is the atom aromatic? (0 or 1)
        int(atom.IsInRing()),              # Is the atom in a ring? (0 or 1)
        atom.GetTotalNumHs(),              # Total number of attached hydrogens
    ])


def show_mol(mol, idx):
    from rdkit.Chem import Draw
    from PIL import Image, ImageDraw, ImageFont
    # Add atom indices to the molecule for visualization
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    # Generate and display the annotated molecule image
    img = Draw.MolToImage(mol, size=(800, 800))
    # Convert to PIL image for annotation
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Set font (adjust path if necessary)
    try:
        font = ImageFont.truetype("arial.ttf", 50)  # Windows / Default
    except:
        font = ImageFont.load_default()  # Fallback if font is missing

    # Add the `idx` text in the top-left corner
    draw.text((30, 30), f"Idx: {idx}", fill="red", font=font)

    img.show()


def smiles_to_graph_with_labels(smiles, idx):
    """
    Convert an RDKit molecule to an adjacency matrix that includes bond multiplicity.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        adj_matrix (np.ndarray): Adjacency matrix with bond orders.
        feature_matrix (np.ndarray): Node feature matrix (atomic numbers).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[!] Invalid SMILES at index {idx}: {smiles}")
        return None, None, None  # or handle however your pipeline expects
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"[!] Sanitization failed at index {idx}: {smiles}\nReason: {e}")
        return None, None, None

    num_atoms = mol.GetNumAtoms()
    # print(smiles)

    # show_mol(mol, idx)
    # Initialize adjacency matrix with zeros
    # adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    # # Initialize node feature matrix (first column = atomic number)
    # feature_matrix = np.zeros((num_atoms, 1), dtype=int)
    # # Extract atomic numbers for each atom
    # for atom in mol.GetAtoms():
    #     idx = atom.GetIdx()
    #     feature_matrix[idx, 0] = atom.GetAtomicNum()  # First feature is atomic number
    #
    # Initialize adjacency matrix
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    # Fill adjacency matrix with bond types
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        # Assign bond type in adjacency matrix
        if bond_type == Chem.rdchem.BondType.SINGLE:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Ensure symmetry
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            adj_matrix[i, j] = 2
            adj_matrix[j, i] = 2
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            adj_matrix[i, j] = 3
            adj_matrix[j, i] = 3
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            adj_matrix[i, j] = 4  # Encoding for aromatic bonds
            adj_matrix[j, i] = 4
    # print("adj_matrix")
    # print(adj_matrix)
    ele0_list = []
    ele1_list = []
    ele2_list = []
    ele3_list = []
    ele4_list = []
    ele5_list = []
    ele6_list = []

    # Atom features (as vectors)
    for atom in mol.GetAtoms():
        node_elements = (atom_to_feature_vector(atom))
        ele0_list.append(int(node_elements[0]))
        ele1_list.append(node_elements[1])
        ele2_list.append(node_elements[2])
        ele3_list.append(node_elements[3])
        ele4_list.append(node_elements[4])
        ele5_list.append(node_elements[5])
        ele6_list.append(node_elements[6])

    feature_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    atom_features = np.array(feature_list)
    # print("atom_features")
    # print(atom_features)
    atom_labels = np.array([assign_node_classes(atom) for atom in mol.GetAtoms()])

    # Find rows where the sum is zero
    zero_sum_rows = np.where(adj_matrix.sum(axis=1) == 0)[0]

    # Print corresponding rows from attr_matrix
    # if len(zero_sum_rows) > 0:
    #     print(f"{idx} - Rows in attr_matrix corresponding to zero-sum rows in adj_matrix:")
    #     print(f"{smiles}")
    #     print(atom_features[zero_sum_rows])

    return adj_matrix, atom_features, atom_labels, ele0_list, ele1_list, ele2_list, ele3_list, ele4_list, ele5_list, ele6_list

# Collect all adjacency matrices, feature matrices, and atom labels
adj_matrices = []
attr_matrices = []
node_labels = []
ele0_list_list = []
ele1_list_list = []
ele2_list_list = []
ele3_list_list = []
ele4_list_list = []
ele5_list_list = []
ele6_list_list = []


def print_element_stats(mylist, name):
    frequency = Counter(mylist)
    for element, count in frequency.items():
        print(name)
        print(f"Element: {element}, Frequency: {count}")

# Batch storage
adj_matrices = []
attr_matrices = []
batch_size = 1000
batch_idx = 0
save_dir = "/Users/taka/Documents/additional_data_for_analysis/"
smiles_sub_list = []


def pad_adj_matrix(matrix, max_size):
    """Pads a square adjacency matrix with zeros to make it (max_size, max_size)."""
    padded = np.zeros((max_size, max_size), dtype=matrix.dtype)
    padded[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded


def pad_attr_matrix(matrix, max_size):
    """Pads a square adjacency matrix with zeros to make it (max_size, max_size)."""
    padded = np.zeros((max_size, 7), dtype=matrix.dtype)
    padded[:matrix.shape[0], :] = matrix
    return padded


max_size = 100

count_sub = 0
for idx, smiles in enumerate(smiles_list):
    # if idx == 4:
    #     break
    adj_matrix, atom_features, atom_labels, *_ = smiles_to_graph_with_labels(smiles, idx)
    # if idx < 3:
    #     print(adj_matrix[:20, :20])
    #     print(smiles)
    # 大きな分子は除く
    if adj_matrix.shape[0] >= 100:
        continue
    else:
        count_sub += 1
    adj_matrices.append(adj_matrix)
    attr_matrices.append(atom_features)
    smiles_sub_list.append(smiles)
    # Process batch every 1000 iterations
    if count_sub % batch_size == 0 or idx + 1 == len(smiles_list):
        # print("Concatenate matrices")
        adj_matrices_padded = [pad_adj_matrix(mat, max_size) for mat in adj_matrices]
        concatenated_adj = np.concatenate(adj_matrices_padded, axis=0)  # Adjust axis if needed
        attr_matrices_padded = [pad_attr_matrix(mat, max_size) for mat in attr_matrices]
        concatenated_attr = np.concatenate(attr_matrices_padded, axis=0)  # Adjust axis if needed
        # print(concatenated_adj.shape)
        # Save batch
        np.save(f"{save_dir}/concatenated_adj_batch_{batch_idx}.npy", concatenated_adj)
        np.save(f"{save_dir}/concatenated_attr_batch_{batch_idx}.npy", concatenated_attr)
        with open(f"{save_dir}/smiles_{batch_idx}.txt", "w") as f:
            for item in smiles_sub_list:
                f.writelines(f"{item}\n")
        # Reset lists for the next batch
        adj_matrices = []
        attr_matrices = []
        smiles_sub_list = []
        count_sub = 0
        batch_idx += 1




