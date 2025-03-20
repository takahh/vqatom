import torch
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point2D
from rdkit.Chem.Draw import rdMolDraw2D

# Convert the binary drawing to an image
from PIL import Image
from io import BytesIO

# Mapping atomic numbers to element symbols
element_symbols = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

# Function to get atomic element symbol
def get_element_symbol(atomic_number):
    return element_symbols.get(int(atomic_number), '?')  # Default to '?' if not found

# Data
src = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    19, 20, 21, 22, 23, 24, 24, 25, 25, 26, 26, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 36, 37, 37, 38, 38])

dst = torch.tensor([0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 16, 17,
                    18, 19, 19, 21, 22, 21, 23, 10, 17, 8, 25, 26, 27, 28, 29, 30,
                    31, 32, 32, 32, 31, 28, 36, 3, 7])

features = torch.tensor([
    [6., 1., 0., 4., 0., 0., 3.],
    [6., 3., 0., 4., 0., 0., 1.],
    [6., 1., 0., 4., 0., 0., 3.],
    [6., 3., 0., 3., 1., 1., 0.],
    [6., 2., 0., 3., 1., 1., 1.],
    [6., 2., 0., 3., 1., 1., 1.],
    [7., 2., 0., 3., 1., 1., 0.],
    [6., 3., 0., 3., 1., 1., 0.],
    [6., 3., 0., 3., 1., 1., 0.],
    [7., 2., 0., 3., 1., 1., 0.],
    [6., 3., 0., 3., 1., 1., 0.],
    [6., 2., 0., 3., 1., 1., 1.],
    [6., 3., 0., 3., 1., 1., 0.],
    [6., 3., 0., 3., 0., 0., 0.],
    [8., 1., 0., 3., 0., 0., 0.],
    [8., 1., 0., 3., 0., 0., 1.],
    [7., 2., 0., 3., 1., 1., 0.],
    [6., 3., 0., 3., 1., 1., 0.],
    [7., 2., 0., 3., 0., 0., 1.],
    [6., 3., 0., 4., 0., 0., 1.],
    [6., 1., 0., 4., 0., 0., 3.],
    [6., 3., 0., 4., 0., 1., 1.],
    [6., 2., 0., 4., 0., 1., 2.],
    [6., 2., 0., 4., 0., 1., 2.],
    [6., 2., 0., 4., 0., 1., 2.],
])

# Create an RDKit molecule
mol = Chem.RWMol()
atom_mapping = {}  # Map original node index to RDKit atom index
atom_labels = {}  # For custom atom labels

# Add atoms and annotate with class labels
for idx, feature_vector in enumerate(features):
    atomic_num = int(feature_vector[0])  # First feature is atomic number
    atom = Chem.Atom(atomic_num)
    atom_idx = mol.AddAtom(atom)
    atom_mapping[idx] = atom_idx

    # Annotate atom with class number
    element = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
    atom_labels[atom_idx] = f"{element}{idx}"  # Add class number

# Define a bond type map
bond_type_map = {1: Chem.BondType.SINGLE,
                 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE,
                 4: Chem.BondType.AROMATIC}

# Add bonds
for s, d in zip(src, dst):
    s, d = int(s), int(d)
    if s in atom_mapping and d in atom_mapping:
        mol.AddBond(atom_mapping[s], atom_mapping[d], Chem.BondType.SINGLE)

# Compute 2D coordinates for drawing
AllChem.Compute2DCoords(mol)

# Prepare the molecule for drawing with kekulization disabled
mol_for_drawing = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)

# Draw the molecule with atom labels
drawer = rdMolDraw2D.MolDraw2DCairo(1000, 400)
drawer.drawOptions().addAtomIndices = True  # Show atom indices
drawer.DrawMolecule(mol)
drawer.FinishDrawing()

# Convert to an image and save
img = Image.open(BytesIO(drawer.GetDrawingText()))
img.show()
img.save("molecule.png")
