#!/usr/bin/env python3
"""Patch local VQAtom to add deterministic 16D semantic hop aggregation."""
from pathlib import Path
import sys, shutil, datetime
root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv)>1 else Path.cwd()
train = root/'quantize'/'vqatom_train.py'
launcher = root/'quantize'/'launch_quantization.py'
if not train.is_file() or not launcher.is_file():
    sys.exit('ERROR: run from vqatom root, or pass the repository path')
s=train.read_text()
old='choices=["legacy", "separate"]'
if '"integrated16"' in s:
    sys.exit('ERROR: integrated16 already exists; no changes made')
assert s.count(old)==1, 'unexpected hop-mode parser layout'
assert s.count('self.semantic_feat = SemanticChemicalAtomFeatures()')==1
assert s.count('if self.semantic_hop_mode == "separate":')==1
assert 'self.semantic_hop1_weight * m1' in s
assert 'self.semantic_hop2_weight * m2' in s
l=launcher.read_text()
assert l.count('--semantic_atom_features --semantic_hop_mode separate')==1
assert l.count('--semantic_hop1_weight 1.0 --semantic_hop2_weight 1.0')==1
s=s.replace(old,'choices=["legacy", "separate", "integrated16"]',1)
s=s.replace('self.semantic_feat = SemanticChemicalAtomFeatures()', '''self.semantic_feat = SemanticChemicalAtomFeatures()
        # Fixed random orthonormal projection: 49D semantic features -> 16D.
        # The same projection is applied to center, exact-hop1, and exact-hop2.
        # Seed is fixed independently of experiment seed for comparability.
        if self.semantic_atom_features and self.semantic_hop_mode == "integrated16":
            gen = torch.Generator(device="cpu").manual_seed(20261103)
            mat = torch.randn(self.semantic_feat.out_dim, 16, generator=gen)
            proj, _ = torch.linalg.qr(mat, mode="reduced")
            self.register_buffer("semantic_proj16", proj.contiguous())''',1)
s=s.replace('self.semantic_hop_mode == "separate" else\n                       self.semantic_feat.out_dim if self.semantic_atom_features else args.hidden_dim)', 'self.semantic_hop_mode == "separate" else\n                       16 if self.semantic_atom_features and self.semantic_hop_mode == "integrated16" else\n                       self.semantic_feat.out_dim if self.semantic_atom_features else args.hidden_dim)',1)
assert '16 if self.semantic_atom_features and self.semantic_hop_mode == "integrated16"' in s
s=s.replace('''if self.semantic_hop_mode == "separate":
                m1, m2 = semantic_exact_hop_means(h_vq, src, dst)''','''if self.semantic_hop_mode == "integrated16":
                # Calculate exact graph-distance hop means in 49D, then project
                # all three components into one shared 16D coordinate system.
                m1, m2 = semantic_exact_hop_means(h_vq, src, dst)
                p = self.semantic_proj16.to(device=h_vq.device, dtype=h_vq.dtype)
                h_vq = (h_vq + self.semantic_hop1_weight * m1
                        + self.semantic_hop2_weight * m2) @ p
            elif self.semantic_hop_mode == "separate":
                m1, m2 = semantic_exact_hop_means(h_vq, src, dst)''',1)
l=l.replace('--semantic_atom_features --semantic_hop_mode separate', '--semantic_atom_features --semantic_hop_mode integrated16',1)
# Avoid collision with prior 147D artifacts. Replace label only where it occurs.
if 'semantic147_w1' in l:
    l=l.replace('semantic147_w1','semantic16_integrated_w1')
# If output labels are dynamic, show warning to check them.
if 'semantic147_w1' not in launcher.read_text():
    print('NOTE: verify output/remote filenames do not overwrite existing 147D artifacts')
compile(s, str(train), 'exec'); compile(l, str(launcher), 'exec')
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for path in (train, launcher):
    backup=path.with_name(path.name+'.bak_'+stamp)
    shutil.copy2(path,backup)
    print('Backup:',backup)
train.write_text(s)
launcher.write_text(l)
print('SUCCESS: integrated16 added; 49D -> fixed 16D projection, hop weights 1.0')
print('Launch with existing launch_quantization.py arguments; inspect diff before running')
