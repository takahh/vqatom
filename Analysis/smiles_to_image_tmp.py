from rdkit import Chem
from rdkit.Chem import Draw

from Analysis.parse_outputs import namelist

indole_smiles = [
    "c1cc2ccccc2[nH]1",              # Indole
    "c1cc2c(cc1)cc[nH]2",            # 2-Methylindole
    "c1ccc2c(c1)[nH]cc2C",           # 3-Methylindole
    "COc1ccc2[nH]ccc2c1",            # 5-Methoxyindole
    "c1ccc2c(c1)cnc2",               # Isoindole
    "c1ccc2[nH]c(C)cc2c1",           # 1-Methylindole
    "c1ccc2[nH]ccc2c1O",             # 5-Hydroxyindole
    "c1cc2[nH]c(C)c(C)cc2c1",        # 2,3-Dimethylindole
    "c1ccc2[nH]ccc2c1CO",            # Indole-3-methanol
    "c1ccc2[nH]c(C(=O)O)cc2c1",      # Indole-3-acetic acid
]

amine_smiles = [
    'CC(=O)Nc1ccc(O)cc1 ',          # Paracetamol (Acetaminophen)
    'Nc1ccc(O)cc1',                 # 4-Aminophenol
    'CC(=O)Nc1ccccc1N',             # o-Aminoacetanilide
    'CC(=O)Nc1ccc(NC)cc1',          # N-Methyl-4-aminoacetanilide
    'CC(=O)Nc1ccc(NC(C)=O)cc1',     # N-Acetyl-4-aminoacetanilide
    'Cc1ccc(N)cc1',                 # 4-Methylaniline
    'CC(=O)Nc1cc(N)ccc1',           # m-Aminoacetanilide
    'Nc1ccc(N)cc1',                # Benzene-1,2-diamine
    'CC(=O)Nc1ccccc1',            # Acetanilide
    'Cc1ccc(N)cc1N'                # 2,5-Diaminotoluene
]

pyridine_smiles = [
    "C1=CC=NC=C1",                    # Nicotine (pyridine part)
    "c1ccncc1",                       # Pyridine
    "c1ccncc1C",                      # Methylpyridine (picoline)
    "c1cc(C)cnc1",                    # 3-Methylpyridine
    "c1cnccc1Cl",                     # 2-Chloropyridine
    "c1ccncc1O",                      # Hydroxypyridine
    "c1cc(CN)cnc1",                   # 2-Aminomethylpyridine
    "c1ccncc1C(=O)O",                 # Nicotinic acid
    "c1ccncc1C(=O)NC",                # Nicotinamide
    "c1cnccc1N",                      # 2-Aminopyridine
    "c1ccncc1C(=O)OC"                 # Pyridinecarboxylic ester
]
phenyl_smiles = [
    "CC(=O)Nc1ccc(O)cc1",             # Paracetamol
    "c1ccccc1",                       # Benzene
    "Cc1ccccc1",                      # Toluene
    "COc1ccccc1",                     # Anisole
    "CC(=O)c1ccccc1",                 # Acetophenone
    "Oc1ccccc1",                      # Phenol
    "Nc1ccccc1",                      # Aniline
    "Clc1ccccc1",                     # Chlorobenzene
    "c1ccc(cc1)C#N",                  # Benzonitrile
    "CC(=O)Oc1ccccc1",                # Phenyl acetate
    "c1ccc(cc1)N(=O)=O"               # Nitrobenzene
]

carbonyl_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
    "CC(=O)OC",                       # Methyl acetate
    "CC(=O)C",                        # Acetone
    "CC(=O)O",                        # Acetic acid
    "CC(=O)N",                        # Acetamide
    "CC(=O)c1ccccc1",                 # Acetophenone
    "O=CC1=CC=CC=C1",                 # Benzaldehyde
    "O=C1CCCCC1",                     # Cyclohexanone
    "CC(=O)OC(C)C",                   # Isopropyl acetate
    "CC(=O)N(C)C",                    # Dimethylacetamide (DMA)
    "O=Cc1ccc(O)cc1"                  # Salicylaldehyde
]

hydroxyl_smiles = [
    "CC(C)Cc1ccc(O)cc1O",             # Propofol
    "Oc1ccccc1",                      # Phenol
    "CC(C)C(O)c1ccccc1",              # Carvacrol
    "COc1ccccc1O",                    # Guaiacol
    "OC(C)c1ccccc1",                  # 1-Phenylethanol
    "CC(O)c1ccc(O)cc1",               # Thymol
    "CC(O)C(C)(C)O",                  # Tert-butyl alcohol
    "CC(C)(O)c1ccccc1",               # p-tert-butylphenol
    "CC(O)CO",                        # Propylene glycol
    "OCC(O)CO",                       # Glycerol
    "CC(O)c1ccc(O)cc1C"               # 2,4-Dimethylresorcinol
]
smileslist = [indole_smiles, amine_smiles, pyridine_smiles, phenyl_smiles, carbonyl_smiles, hydroxyl_smiles]
namelist = ['indole_smiles', 'amine_smiles', 'pyridine_smiles', 'phenyl_smiles', 'carbonyl_smiles', 'hydroxyl_smiles']
for idx, smiles_list in enumerate(smileslist):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400,400))
    img.show()
    img.save(f'/Users/taka/Documents/similar_molecules/{namelist[idx]}.png')
