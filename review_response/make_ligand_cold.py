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
    assignments={'original':df._original_split.copy()}
    summary=raw_summary+[analyze(df,'combined')]
    for name in NAMES:
        summary.append(analyze(df.loc[df._original_split==name],'cleaned_original_'+name))
    for iteration in (1,2,3):
        best=None; best_score=float('inf')
        for _ in range(a.candidates):
            perm=rng.permutation(len(keys)); assign=np.zeros(len(keys),dtype=np.int8)
            targets=[int(total*.1),int(total*.1)]; sizes=[0,0]
            for ix in perm:
                choices=[j for j in range(2) if sizes[j]<targets[j]]
                if choices:
                    j=choices[int(rng.integers(len(choices)))]; assign[ix]=j+1;sizes[j]+=int(counts[ix])
            if not (assign==1).any() or not (assign==2).any():continue
            sz=np.array([counts[assign==j].sum() for j in range(3)],dtype=float)
            if (sz==0).any():continue
            rates=np.array([positives[assign==j].sum()/sz[j] for j in range(3)])
            row_pen=((sz/total-np.array([.8,.1,.1]))**2).sum()*70
            rate_pen=((rates-global_rate)**2).sum()*4
            test_mask=assign==2
            overlap_pen=sum(int(counts[i]) for i in np.flatnonzero(test_mask) if keys[i] in original_test)/sz[2]
            new_pen=sum(sum(int(counts[i]) for i in np.flatnonzero(test_mask) if keys[i] in prev)/sz[2] for prev in used_tests)
            score=row_pen+rate_pen+2*overlap_pen+2*new_pen
            if score<best_score:best_score=score;best=assign.copy()
        if best is None:raise RuntimeError('No valid split candidate')
        split_map=dict(zip(keys,[NAMES[int(j)] for j in best]))
        assignment=df._ligand.map(split_map)
        assignments[f'split_{iteration}']=assignment
        folder=a.out/f'split_{iteration}';folder.mkdir(exist_ok=True)
        for name in NAMES:
            subset=df.loc[assignment==name].drop(columns=['_original_split','_ligand'])
            subset.to_csv(folder/f'{name}.csv',index=False)
            summary.append(analyze(df.loc[assignment==name],f'split_{iteration}_{name}'))
        pd.DataFrame({'canonical_smiles':keys,'split':[NAMES[int(j)] for j in best]}).to_csv(folder/'ligand_assignment.csv',index=False)
        test_set={keys[i] for i in np.flatnonzero(best==2)}
        print(f'[split {iteration}] score={best_score:.5f} rows={[int((assignment==n).sum()) for n in NAMES]} positive_rates={[round(float((pd.to_numeric(df.loc[assignment==n,"y"])>=12.1).mean()),4) for n in NAMES]} test_overlap_original={len(test_set & original_test)} test_overlap_prior={[len(test_set & prev) for prev in used_tests]}',flush=True)
        used_tests.append(test_set)
    pd.DataFrame(summary).to_csv(a.out/'split_entity_analysis.csv',index=False)
    for r in summary:
        print(f"[analysis] {r['split']}: ligands={r['ligands']} positive_only={r['ligand_positive_only_n']} ({r['ligand_positive_only_pct']:.2f}%) negative_only={r['ligand_negative_only_n']} ({r['ligand_negative_only_pct']:.2f}%) mixed={r['ligand_mixed_n']} ({r['ligand_mixed_pct']:.2f}%)", flush=True)
    sets_by_run={}; report=[]
    for run,assignment in assignments.items():
        sets=original_sets if run=='original' else {name:set(df.loc[assignment==name,'_ligand']) for name in NAMES}
        sets_by_run[run]=sets
        for left,right in [('train','valid'),('train','test'),('valid','test')]:
            common=sets[left]&sets[right]
            report.append(dict(run=run,comparison=f'{left}_vs_{right}',overlap_ligands=len(common),overlap_ids=';'.join(sorted(common))))
            print(f'[coldness] {run} {left}/{right}: {len(common)} overlapping ligands',flush=True)
            if run!='original' and common:raise AssertionError('LigandCold violation')
    pd.DataFrame(report).to_csv(a.out/'ligand_cold_validation.csv',index=False)
    overlap=[]
    for left,right in [('original','split_1'),('original','split_2'),('original','split_3'),('split_1','split_2'),('split_1','split_3'),('split_2','split_3')]:
        aa=sets_by_run[left]['test'];bb=sets_by_run[right]['test'];common=aa&bb
        overlap.append(dict(test_a=left,test_b=right,test_a_ligands=len(aa),test_b_ligands=len(bb),shared_ligands=len(common),
                            shared_pct_of_a=100*len(common)/len(aa) if aa else 0,shared_pct_of_b=100*len(common)/len(bb) if bb else 0,
                            jaccard_pct=100*len(common)/len(aa|bb) if aa|bb else 0,
                            shared_rows_in_a_test=int(((assignments[left]=='test')&df._ligand.isin(common)).sum()) if left!='original' else int(original_test_ligands.isin(common).sum()),
                            shared_rows_in_b_test=int(((assignments[right]=='test')&df._ligand.isin(common)).sum()),
                            shared_ligand_ids=';'.join(sorted(common))))
    pd.DataFrame(overlap).to_csv(a.out/'test_ligand_overlap.csv',index=False)
    print('[done]',a.out.resolve(),flush=True)

if __name__=='__main__':main()
