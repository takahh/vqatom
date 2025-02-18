import numpy as np
import torch
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from scipy.sparse.csgraph import connected_components
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from rdkit.Geometry import Point2D
from PIL import Image
from io import BytesIO

CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 1300
FONTSIZE = 40
EPOCH = 2
PATH = "/Users/taka/Documents/vqgraph_0213/"

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


def compute_molecule_bounds(mol):
    """Calculate the bounding box for a molecule."""
    try:
        conformer = mol.GetConformer()  # Attempt to get the conformer
    except ValueError:
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    minv = Point2D(min(x_coords), min(y_coords))
    maxv = Point2D(max(x_coords), max(y_coords))
    return minv, maxv


def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, classes, node_indices):
    """
    Visualizes molecules with correct bond orders and aromaticity.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Adjacency matrix defining connectivity.
        feature_matrix (numpy.ndarray): Node feature matrix (atomic numbers, hybridization, etc.).
        classes (list): Node classes.

    Returns:
        None: Displays molecule images.
    """
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    images = []

    for i in range(n_components - 2):
        if i == 0:
            continue
        print(f"Processing molecule {i}")

        # Get node indices for this molecule
        component_indices = np.where(labels == i)[0]
        mol_adj = adj_matrix[component_indices, :][:, component_indices]
        mol_features = feature_matrix[component_indices]

        # Create RDKit molecule
        mol = Chem.RWMol()
        atom_map = {}  # Map graph indices to RDKit atom indices

        # Add atoms
        for idx, features in zip(component_indices, mol_features):
            atomic_num = int(features[0])  # Atomic number
            is_aromatic = bool(features[4])  # Aromaticity
            in_ring = bool(features[5])  # Ring membership

            atom = Chem.Atom(atomic_num)
            atom.SetIsAromatic(is_aromatic and in_ring)  # Mark as aromatic if in a ring
            atom_idx = mol.AddAtom(atom)
            atom_map[idx] = atom_idx

        # Add bonds with correct bond order
        bond_type_map = {1: Chem.BondType.SINGLE,
                         2: Chem.BondType.DOUBLE,
                         3: Chem.BondType.TRIPLE,
                         4: Chem.BondType.AROMATIC}

        for x, y in zip(*np.where(mol_adj > 0)):
            if x < y:  # Avoid duplicate bonds
                bond_order = int(mol_adj[x, y])  # Extract bond order
                bond_type = bond_type_map.get(bond_order, Chem.BondType.SINGLE)
                try:
                    mol.AddBond(atom_map[x], atom_map[y], bond_type)
                except KeyError:
                    pass

        # Compute 2D coordinates for proper display
        AllChem.Compute2DCoords(mol)

        # Ensure correct aromatic representation
        # Chem.Kekulize(mol, clearAromaticFlags=True)
        #
        # # Sanitize molecule
        # Chem.SanitizeMol(mol)

        # Draw molecule with custom settings
        drawer = Draw.MolDraw2DCairo(CANVAS_WIDTH, CANVAS_HEIGHT)
        options = drawer.drawOptions()
        options.atomLabelFontSize = FONTSIZE
        options.bondLineWidth = 2
        options.scaleBondWidth = True  # Scale bond width relative to image size

        # Draw the molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Convert binary image data to an image
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        images.append(img)

    # Display images
    for i, img in enumerate(images):
        plt.figure(dpi=150)
        plt.title(f"Molecule {i+1}")
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

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
    adj_file = f"{path}/sample_adj_{EPOCH}.npz"                     # input data
    feat_file = f"{path}sample_node_feat_{EPOCH}.npz"      # assigned code vector id
    indices_file = f"{path}sample_emb_ind_{EPOCH}.npz"
    arr_indices = getdata(indices_file)   # indices of the input
    arr_adj = getdata(adj_file)       # assigned quantized code vec indices
    arr_feat = getdata(feat_file)       # assigned quantized code vec indices
    arr_feat = restore_node_feats(arr_feat)
    node_indices = [int(x) for x in arr_indices.tolist()]
    subset_adj_matrix = arr_adj[0:200, 0:200]
    subset_attr_matrix = arr_feat[:200]
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_attr_matrix, None, node_indices)


if __name__ == '__main__':
    main()