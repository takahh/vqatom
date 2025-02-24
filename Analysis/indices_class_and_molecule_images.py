import numpy as np
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
# np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from scipy.sparse.csgraph import connected_components
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from rdkit.Geometry import Point2D
from rdkit.Chem.Draw import rdMolDraw2D

# Convert the binary drawing to an image
from PIL import Image
from io import BytesIO
import rdkit
print(rdkit.__version__)
from rdkit import Chem
print(Chem.__file__)


CANVAS_WIDTH = 2300
CANVAS_HEIGHT = 1500
FONTSIZE = 40
EPOCH = 13
PATH = "/Users/taka/Documents/vqgraph_0222/"

def getdata(filename):
    # filename = "out_emb_list.npz"
    if "mol" in filename:
        arr = np.load(f"{filename}")
    else:
        arr = np.load(f"{filename}")["arr_0"]
    # arr = np.squeeze(arr)
    return arr

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem


def create_adjacency_matrix(src, dst, num_nodes, directed=True):
    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Fill the matrix based on src and dst
    for s, d in zip(src, dst):
        adj_matrix[s, d] = 1
        if not directed:
            adj_matrix[d, s] = 1

    return adj_matrix


def compute_molecule_bounds(mol):
    """Calculate the bounding box for a molecule."""
    try:
        conformer = mol.GetConformer()  # Attempt to get the conformer
    except ValueError:
        # Fallback: Generate 2D coordinates and retry
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    minv = Point2D(min(x_coords), min(y_coords))
    maxv = Point2D(max(x_coords), max(y_coords))
    return minv, maxv


def to_superscript(number):
    """Convert a number to its Unicode superscript representation."""
    superscript_map = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    }
    return "".join(superscript_map.get(char, char) for char in str(number))

def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, classes, arr_src, arr_dst, arr_bond_order, adj_matrix_base):
    import numpy as np
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    # from rdkit.Chem.Draw import rdMolDraw2D
    import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D

    from scipy.sparse.csgraph import connected_components
    from PIL import Image
    from io import BytesIO

    # Ensure node indices match the feature matrix.
    node_indices = np.arange(feature_matrix.shape[0])
    node_to_class = {node: cls for node, cls in zip(node_indices, classes)}

    # Identify connected components (molecules)
    n_components, labels = connected_components(csgraph=adj_matrix_base, directed=False)
    print("n_components")
    print(n_components)

    images = []
    for i in range(n_components - 2):
        print(f"$$$$$$$$$$$$$$$$$$$. {i}")
        # Get node indices for this molecule
        component_indices = np.where(labels == i)[0]
        # Extract subgraph features for this component
        mol_features = feature_matrix[component_indices]

        # Filter edges to only those within the component
        mask = np.isin(arr_src, component_indices) & np.isin(arr_dst, component_indices)
        mol_src = arr_src[mask]
        mol_dst = arr_dst[mask]
        mol_bond = arr_bond_order[mask]
        print("mol_features:", mol_features)
        print("mol_bond:", mol_bond)
        print("mol_src:", mol_src)
        print("mol_dst:", mol_dst)

        # Create an editable RDKit molecule
        mol = Chem.RWMol()
        atom_mapping = {}  # Map original node index to RDKit atom index
        atom_labels = {}   # For custom atom labels in the drawing

        # Add atoms and annotate with class labels
        for idx, features in zip(component_indices, mol_features):
            atomic_num = int(features[0])  # Assume the first feature is the atomic number
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_mapping[idx] = atom_idx

            # Annotate atom with its class label if available
            class_label = node_to_class.get(idx, "Unknown")
            element = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            atom_labels[atom_idx] = f"{element}{class_label}" if class_label != "Unknown" else element

        # Define a bond type map
        bond_type_map = {1: Chem.BondType.SINGLE,
                         2: Chem.BondType.DOUBLE,
                         3: Chem.BondType.TRIPLE,
                         4: Chem.BondType.AROMATIC}

        # Build a dictionary for unique bonds.
        unique_bonds = {}
        for src, dst, bond_order in zip(mol_src, mol_dst, mol_bond):
            src, dst, bond_order = int(src), int(dst), int(bond_order)
            # Ensure both atoms exist in the mapping.
            if src not in atom_mapping or dst not in atom_mapping:
                continue
            src_mol, dst_mol = atom_mapping[src], atom_mapping[dst]
            # Avoid self-bonds.
            if src_mol == dst_mol:
                continue
            # Use a sorted tuple so bond direction doesn't matter.
            key = tuple(sorted([src_mol, dst_mol]))
            # If the same bond appears multiple times, use the maximum bond order.
            unique_bonds[key] = max(unique_bonds.get(key, 0), bond_order)

        # Now add each unique bond to the molecule.
        for (src_mol, dst_mol), bond_order in unique_bonds.items():
            bond_type = bond_type_map.get(bond_order, Chem.BondType.SINGLE)
            mol.AddBond(src_mol, dst_mol, bond_type)
            # If the bond should be aromatic, update the aromatic flags.
            if bond_order == 4:
                mol.GetAtomWithIdx(src_mol).SetIsAromatic(True)
                mol.GetAtomWithIdx(dst_mol).SetIsAromatic(True)
                bond = mol.GetBondBetweenAtoms(src_mol, dst_mol)
                if bond is not None:
                    bond.SetIsAromatic(True)

        # Compute 2D coordinates for drawing.
        AllChem.Compute2DCoords(mol)

        # Sanitize the molecule without kekulization to preserve aromatic flags.
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as e:
            print(f"Sanitization warning: {e}")

        # Prepare the molecule for drawing with kekulization disabled.
        mol_for_drawing = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)

        # Create a drawing canvas.
        drawer = Draw.MolDraw2DCairo(1500, 1000)
        options = drawer.drawOptions()
        options.atomLabelFontSize = 4  # Increase font size for readability

        # Assign custom labels.
        for idx, label in atom_labels.items():
            options.atomLabels[idx] = label

        # Draw the molecule.
        drawer.DrawMolecule(mol_for_drawing)
        drawer.FinishDrawing()

        # Convert the drawing to an image.
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        images.append(img)

    # Display all images.
    for i, img in enumerate(images):
        plt.figure(dpi=250)
        plt.title(f"Molecule {i + 1}")
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()



import torch
import torch
import numpy as np

def restore_node_feats(transformed):
    # Convert to PyTorch tensor if it's a NumPy array
    if isinstance(transformed, np.ndarray):
        transformed = torch.tensor(transformed, dtype=torch.float32)

    restored = torch.empty_like(transformed, dtype=torch.float32)

    restored[:, 0] = torch.where(transformed[:, 0] == 1, 6,
                      torch.where(transformed[:, 0] == 20, 8,
                      torch.where(transformed[:, 0] == 10, 7,
                      torch.where(transformed[:, 0] == 5, 17,
                      torch.where(transformed[:, 0] == 15, 9,
                      torch.where(transformed[:, 0] == 8, 35,
                      torch.where(transformed[:, 0] == 3, 16,
                      torch.where(transformed[:, 0] == 12, 15,
                      torch.where(transformed[:, 0] == 18, 1,
                      torch.where(transformed[:, 0] == 2, 5,
                      torch.where(transformed[:, 0] == 16, 53,
                      torch.where(transformed[:, 0] == 4, 14,
                      torch.where(transformed[:, 0] == 6, 34,
                      torch.where(transformed[:, 0] == 7, 19,
                      torch.where(transformed[:, 0] == 9, 11,
                      torch.where(transformed[:, 0] == 11, 3,
                      torch.where(transformed[:, 0] == 13, 30,
                      torch.where(transformed[:, 0] == 14, 33,
                      torch.where(transformed[:, 0] == 17, 12,
                      torch.where(transformed[:, 0] == 19, 52, -2))))))))))))))))))))

    restored[:, 1] = torch.where(transformed[:, 1] == 1, 1,
                      torch.where(transformed[:, 1] == 20, 2,
                      torch.where(transformed[:, 1] == 10, 3,
                      torch.where(transformed[:, 1] == 15, 0,
                      torch.where(transformed[:, 1] == 5, 4,
                      torch.where(transformed[:, 1] == 7, 6,
                      torch.where(transformed[:, 1] == 12, 5, -2)))))))

    restored[:, 2] = torch.where(transformed[:, 2] == 1, 0,
                      torch.where(transformed[:, 2] == 20, 1,
                      torch.where(transformed[:, 2] == 10, -1,
                      torch.where(transformed[:, 2] == 5, 3,
                      torch.where(transformed[:, 2] == 15, 2, -2)))))

    restored[:, 3] = torch.where(transformed[:, 3] == 1, 4,
                      torch.where(transformed[:, 3] == 20, 3,
                      torch.where(transformed[:, 3] == 10, 1,
                      torch.where(transformed[:, 3] == 5, 2,
                      torch.where(transformed[:, 3] == 15, 7,
                      torch.where(transformed[:, 3] == 18, 6, -2))))))

    restored[:, 4] = torch.where(transformed[:, 4] == 1, 0,
                      torch.where(transformed[:, 4] == 20, 1, -2))

    restored[:, 5] = torch.where(transformed[:, 5] == 1, 0,
                      torch.where(transformed[:, 5] == 20, 1, -2))

    restored[:, 6] = torch.where(transformed[:, 6] == 1, 3,
                      torch.where(transformed[:, 6] == 20, 0,
                      torch.where(transformed[:, 6] == 10, 1,
                      torch.where(transformed[:, 6] == 15, 2,
                      torch.where(transformed[:, 6] == 5, 4, -2)))))

    return restored.numpy()  # Convert back to NumPy array if needed


def main():
    path = PATH
    adj_file = f"{path}/sample_adj_{EPOCH}.npz"
    adj_base_file = f"{path}/sample_adj_base_{EPOCH}.npz"                     # input data
    feat_file = f"{path}sample_node_feat_{EPOCH}.npz"      # assigned code vector id
    indices_file = f"{path}sample_emb_ind_{EPOCH}.npz"
    bond_order_file = f"{path}sample_bond_num_{EPOCH}.npz"
    src_file = f"{path}sample_src_{EPOCH}.npz"
    dst_file = f"{path}sample_dst_{EPOCH}.npz"

    arr_indices = getdata(indices_file)   # indices of the input
    arr_adj = getdata(adj_file)       # assigned quantized code vec indices
    arr_adj_base = getdata(adj_base_file)       # assigned quantized code vec indices
    arr_feat = getdata(feat_file)       # assigned quantized code vec indices
    arr_feat = restore_node_feats(arr_feat)
    node_indices = [int(x) for x in arr_indices.tolist()]
    arr_src = getdata(src_file)
    arr_dst = getdata(dst_file)
    arr_bond_order = getdata(bond_order_file)

    # -------------------------------------
    # rebuild attr matrix
    # -------------------------------------
    # attr_data = arr_input["attr_data"]
    # attr_indices = arr_input["attr_indices"]
    # attr_indptr = arr_input["attr_indptr"]
    # attr_shape = arr_input["attr_shape"]
    # attr_matrix = csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
    # ic(node_indices[0])
    # subset_attr_matrix = attr_matrix[node_indices[0]:node_indices[0] + 200, :].toarray()
    # subset_attr_matrix = attr_matrix.toarray()

    # -------------------------------------
    # rebuild adj matrix
    # -------------------------------------
    # Assuming you have these arrays from your input
    # adj_data = arr_input["adj_data"]
    # adj_indices = arr_input["adj_indices"]
    # adj_indptr = arr_input["adj_indptr"]
    # adj_shape = arr_input["adj_shape"]
    # Reconstruct the sparse adjacency matrix
    # adj_matrix = csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
    arr_adj = arr_adj[0:300, 0:300]
    subset_adj_matrix = arr_adj[0:300, 0:300]
    subset_adj_base_matrix = arr_adj_base[0:300, 0:300]
    subset_attr_matrix = arr_feat[:300]
    print(f"arr_adj {arr_adj.shape}")
    print(f"subset_adj_base_matrix {subset_adj_base_matrix.shape}")
    # -------------------------------------
    # split the matrix into molecules
    # -------------------------------------
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_attr_matrix, node_indices, arr_src, arr_dst, arr_bond_order, subset_adj_base_matrix)


if __name__ == '__main__':
    main()