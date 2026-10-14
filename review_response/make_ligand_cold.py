#!/usr/bin/env python3
"""Create three ligand-cold KIBA splits from existing train/valid/test CSVs.
Requires: pandas numpy rdkit. Group by canonical isomeric SMILES.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

NAMES = ('train', 'valid', 'test')

def canonicalize(s):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {str(s)[:120]}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def analyze(df, name):
    labels = (pd.to_numeric(df.y) >= 12.1).astype(int)
    lig_partners = df.groupby('_ligand')['seq'].nunique()
    prot_partners = df.groupby('seq')['_ligand'].nunique()
    per_ligand = df.assign(_label=labels).groupby('_ligand')['_label'].agg(['min', 'max'])
    positive_only = (per_ligand['min'] == 1)
    negative_only = (per_ligand['max'] == 0)
    mixed = ~(positive_only | negative_only)
    n = len(per_ligand)
    pct = lambda count: 100.0 * count / n if n else 0.0
    pos_n, neg_n, mixed_n = int(positive_only.sum()), int(negative_only.sum()), int(mixed.sum())
    return dict(split=name, rows=len(df), positive_rows=int(labels.sum()),
                negative_rows=int((labels == 0).sum()), positive_rate=float(labels.mean()),
                ligands=n, proteins=len(prot_partners),
                ligand_positive_only_n=pos_n, ligand_positive_only_pct=pct(pos_n),
                ligand_negative_only_n=neg_n, ligand_negative_only_pct=pct(neg_n),
                ligand_mixed_n=mixed_n, ligand_mixed_pct=pct(mixed_n),
                ligand_one_class_n=pos_n + neg_n, ligand_one_class_pct=pct(pos_n + neg_n),
                ligand_one_partner_n=int(lig_partners.eq(1).sum()),
                ligand_one_partner_pct=float(lig_partners.eq(1).mean()*100),
                ligand_one_class_multiple_partners_n=int(((positive_only | negative_only) & lig_partners.gt(1)).sum()),
                protein_one_partner_pct=float(prot_partners.eq(1).mean()*100))

def main():
    p=argparse.ArgumentParser()
    for name in NAMES:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--out',type=Path,default=Path('kiba_ligand_cold_splits'))
    p.add_argument('--candidates',type=int,default=20000)
    p.add_argument('--seed',type=int,default=20261013)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for name in NAMES:
        frame=pd.read_csv(getattr(a,name),dtype=str,keep_default_na=False)
        missing={'seq','smiles','y'}-set(frame.columns)
        if missing: raise ValueError(f'{name}: missing columns {missing}')
        frame['_original_split']=name
        frames.append(frame)
    df=pd.concat(frames,ignore_index=True)
    if df[['seq','smiles','y']].eq('').any().any(): raise ValueError('Empty seq, smiles or y')
    df['_ligand']=df.smiles.map(canonicalize)
    original_sets={name:set(df.loc[df._original_split==name,'_ligand']) for name in NAMES}
    original_test=original_sets['test']
    original_test_ligands=df.loc[df._original_split=='test','_ligand'].copy()
    raw_summary=[analyze(df.loc[df._original_split==name], 'raw_original_'+name) for name in NAMES]
    raw_counts={name:int((df._original_split==name).sum()) for name in NAMES}
    print(f'[input] raw_rows={raw_counts} original_test_ligands={len(original_test)}',flush=True)
    df['_y_numeric']=pd.to_numeric(df.y,errors='coerce')
    if df._y_numeric.isna().any(): raise ValueError('Non-numeric y')
    df['_label']=(df._y_numeric>=12.1).astype('int8')
    # Group chemically identical ligands, including alternative SMILES representations.
    grouped=df.groupby(['seq','_ligand'],sort=False,dropna=False)
    audit=[]
    for (seq,ligand),part in grouped:
        if len(part)>1:
            audit.append(dict(seq=seq,canonical_smiles=ligand,n_rows=len(part),
                y_values=';'.join(part.y.astype(str)),
                labels=';'.join(map(str,sorted(part._label.unique()))),
                decision='drop_conflicting_labels' if part._label.nunique()>1 else 'median_y'))
    pd.DataFrame(audit,columns=['seq','canonical_smiles','n_rows','y_values','labels','decision']).to_csv(a.out/'duplicate_pair_audit.csv',index=False)
    label_counts=grouped['_label'].nunique()
    medians=grouped['_y_numeric'].median()
    dedup=grouped.head(1).copy()
    idx=pd.MultiIndex.from_frame(dedup[['seq','_ligand']])
    keep=label_counts.reindex(idx).to_numpy()==1
    dedup['y']=[format(v,'.12g') for v in medians.reindex(idx).to_numpy()]
    df=dedup.loc[keep].copy().reset_index(drop=True)
    df.drop(columns=['_y_numeric','_label'],inplace=True)
    print(f'[dedup] repeated_pairs={len(audit)} conflicting_pairs_removed={int((label_counts>1).sum())} remaining_rows={len(df)}',flush=True)
    groups=df.groupby('_ligand',sort=True).indices
    keys=list(groups)
    counts=np.array([len(groups[k]) for k in keys],dtype=np.int64)
    y=pd.to_numeric(df.y).to_numpy()
    positives=np.array([(y[groups[k]]>=12.1).sum() for k in keys],dtype=np.int64)
    total=len(df); global_rate=positives.sum()/total
    rng=np.random.default_rng(a.seed)
    used_tests=[]