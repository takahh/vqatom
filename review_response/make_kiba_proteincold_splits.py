#!/usr/bin/env python3
"""Create two distinct MMseqs40 ProteinCold KIBA splits from existing CSVs.
Requires: pip install pandas; MMseqs2 executable in PATH.
"""
import argparse, hashlib, subprocess, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def analyze_split(df, name):
    # Partners are unique SMILES / protein sequences, not row counts.
    prot_partners = df.groupby('seq')['smiles'].nunique()
    lig_partners = df.groupby('smiles')['seq'].nunique()
    labels = (pd.to_numeric(df['y']) >= 12.1).astype(int)
    annotated = df.assign(_label=labels)
    lig_classes = annotated.groupby('smiles')['_label'].nunique()
    prot_classes = annotated.groupby('seq')['_label'].nunique()
    ligand_one_class = lig_classes.eq(1)
    ligand_one_partner = lig_partners.eq(1)
    result = {
        'split': name, 'rows': len(df), 'positive_rate': float(labels.mean()),
        'proteins': len(prot_partners),
        'protein_one_partner_n': int(prot_partners.eq(1).sum()),
        'protein_one_partner_pct': float(prot_partners.eq(1).mean()*100),
        'protein_one_class_n': int(prot_classes.eq(1).sum()),
        'protein_one_class_pct': float(prot_classes.eq(1).mean()*100),
        'ligands': len(lig_partners),
        'ligand_one_partner_n': int(ligand_one_partner.sum()),
        'ligand_one_partner_pct': float(ligand_one_partner.mean()*100),
        'ligand_one_class_n': int(ligand_one_class.sum()),
        'ligand_one_class_pct': float(ligand_one_class.mean()*100),
        'ligand_one_class_multiple_partners_n': int((ligand_one_class & ~ligand_one_partner).sum()),
        'mean_protein_partners': float(prot_partners.mean()),
        'mean_ligand_partners': float(lig_partners.mean()),
    }
    return result

def save_analysis(df, folder):
    records = [analyze_split(df, 'combined')]
    for name in ('train','valid','test'):
        records.append(analyze_split(df.loc[df['_original_split']==name], 'cleaned_original_'+name))
    for i in (2,3):
        mapping_file=folder/f'split_{i}'/'cluster_assignment.csv'
        if not mapping_file.exists(): continue
        mapping=pd.read_csv(mapping_file, dtype=str)
        if mapping['protein_cluster'].duplicated().any(): raise ValueError('Duplicate cluster assignments')
        split_map=dict(zip(mapping['protein_cluster'],mapping['split']))
        assigned=df['_cluster'].map(split_map)
        if assigned.isna().any(): raise ValueError('Missing cluster assignments')
        for name in ('train','valid','test'):
            records.append(analyze_split(df.loc[assigned==name], f'split_{i}_'+name))
    pd.DataFrame(records).to_csv(folder/'split_entity_analysis.csv',index=False)
    print('[analysis] saved:',folder/'split_entity_analysis.csv',flush=True)
    for r in records:
        print(f"[analysis] {r['split']}: protein_one_partner={r['protein_one_partner_pct']:.2f}% ligand_one_partner={r['ligand_one_partner_pct']:.2f}% ligand_one_class={r['ligand_one_class_pct']:.2f}%",flush=True)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--train',type=Path,required=True);p.add_argument('--valid',type=Path,required=True);p.add_argument('--test',type=Path,required=True)
 p.add_argument('--out',type=Path,default=Path('kiba_extra_splits'))
 p.add_argument('--candidates',type=int,default=20000)
 p.add_argument('--seed',type=int,default=20261013)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if not shutil.which('mmseqs'): p.error('MMseqs2 not found in PATH; install it first')
 frames=[]
 for name,path in [('train',a.train),('valid',a.valid),('test',a.test)]:
  df=pd.read_csv(path,keep_default_na=False,dtype=str);df['_original_split']=name;frames.append(df)
 df=pd.concat(frames,ignore_index=True)
 if df[['seq','smiles']].eq('').any().any():raise ValueError('Empty seq or smiles')
 # Deduplicate pairs BEFORE clustering/splitting. Drop label-discordant pairs,
 # and collapse concordant repeated measurements to a single median-y row.
 df['_y_numeric']=pd.to_numeric(df['y'],errors='coerce')
 if df['_y_numeric'].isna().any(): raise ValueError('Non-numeric y values found')
 df['_label']=(df['_y_numeric']>=12.1).astype('int8')
 grouped=df.groupby(['seq','smiles'],sort=False,dropna=False)
 counts=grouped.size()
 label_counts=grouped['_label'].nunique()
 conflict_keys=label_counts[label_counts>1].index
 repeated_keys=counts[counts>1].index
 audit=[]
 for (seq,smiles),part in grouped:
  if len(part)>1:
   audit.append({'seq':seq,'smiles':smiles,'n_rows':len(part),
                 'y_values':';'.join(part['y'].astype(str)),
                 'labels':';'.join(map(str,sorted(part['_label'].unique()))),
                 'decision':'drop_conflicting_labels' if part['_label'].nunique()>1 else 'median_y'})
 pd.DataFrame(audit,columns=['seq','smiles','n_rows','y_values','labels','decision']).to_csv(a.out/'duplicate_pair_audit.csv',index=False)
 # One representative record per pair, with its y replaced by the median.
 # For repeated pairs spanning original splits, retain first occurrence's origin
 # only for descriptive statistics; original CSVs themselves remain unchanged.
 df=grouped.head(1).copy()
 median_y=grouped['_y_numeric'].median()
 pair_idx=pd.MultiIndex.from_frame(df[['seq','smiles']])
 df['y']=[format(v,'.12g') for v in median_y.reindex(pair_idx).to_numpy()]
 bad_pairs=set(conflict_keys)
 mask=[(seq,smiles) not in bad_pairs for seq,smiles in zip(df['seq'],df['smiles'])]
 df=df.loc[mask].copy()
 df.drop(columns=['_y_numeric','_label'],inplace=True)
 print(f'[dedup] repeated_pairs={len(repeated_keys)} discordant_pairs_removed={len(conflict_keys)} remaining_rows={len(df)}',flush=True)
 if df.duplicated(['seq','smiles']).any(): raise AssertionError('Deduplication failed')
 seqs=sorted(df.seq.unique());fasta=a.out/'protein_unique_seqs.fasta'
 with fasta.open('w') as f:
  for i,s in enumerate(seqs):f.write(f'>seq_{i}\n{s}\n')
 prefix=a.out/'mmseqs40'; tmp=a.out/'mmseqs_tmp'
 tsv=Path(str(prefix)+'_cluster.tsv')
 if not tsv.exists():
  subprocess.run(['mmseqs','easy-cluster',str(fasta),str(prefix),str(tmp),'--min-seq-id','0.4','-c','0.8','--cov-mode','0','--cluster-mode','2','--threads','8'],check=True)
 id_to_seq={f'seq_{i}':s for i,s in enumerate(seqs)};clusters={}
 for line in tsv.read_text().splitlines():
  rep,member=line.split('\t');clusters[id_to_seq[member]]=rep
 if len(clusters)!=len(seqs):raise ValueError('MMseqs output missing sequences')
 df['_cluster']=df.seq.map(clusters)
 pd.DataFrame({'seq':seqs,'protein_cluster':[clusters[s] for s in seqs]}).to_csv(a.out/'seq_to_cluster_40id.csv',index=False)
 # If the original split divides an MMseqs cluster, report rather than silently asserting original ProteinCold.
 original_cluster_sets={s:set(df.loc[df._original_split==s,'_cluster']) for s in ('train','valid','test')}
 original_overlap={f'{x}-{y}':len(original_cluster_sets[x]&original_cluster_sets[y]) for x,y in [('train','valid'),('train','test'),('valid','test')]}
 print('[original cluster overlaps]',original_overlap,flush=True)
 group=df.groupby('_cluster',sort=True)
 keys=list(group.groups); counts=np.array([len(group.groups[k]) for k in keys]); pos=np.array([sum(pd.to_numeric(df.loc[group.groups[k],'y'])>=12.1) for k in keys]);total=len(df);global_rate=pos.sum()/total
 # Optimize ratios and class balance while strongly penalizing original/new test overlap.
 rng=np.random.default_rng(a.seed);selected=[];used_tests=[]
 original_test=set(df.loc[df._original_split=='test','_cluster'])
 for iteration in range(2):
  best=None; best_score=float('inf')
  for _ in range(a.candidates):
   perm=rng.permutation(len(keys)); assign=np.zeros(len(keys),dtype=np.int8)
   # Randomized greedy allocation by row counts; cluster indivisibility preserved.
   targets=[int(total*.1),int(total*.1)];sizes=[0,0]
   for ix in perm:
    choices=[j for j in range(2) if sizes[j]<targets[j]]
    if choices:
     j=choices[int(rng.integers(len(choices)))];assign[ix]=j+1;sizes[j]+=int(counts[ix])
   test_keys={keys[i] for i in np.flatnonzero(assign==2)}
   if not test_keys:continue
   sz=np.array([counts[assign==j].sum() for j in range(3)],dtype=float)
   if (sz==0).any():continue
   rates=np.array([pos[assign==j].sum()/sz[j] for j in range(3)])
   row_pen=((sz/total-np.array([.8,.1,.1]))**2).sum()*70
   rate_pen=((rates-global_rate)**2).sum()*4
   overlap_pen=sum(counts[i] for i in range(len(keys)) if assign[i]==2 and keys[i] in original_test)/max(1,sz[2])
   new_pen=sum(sum(counts[i] for i in range(len(keys)) if assign[i]==2 and keys[i] in prev)/max(1,sz[2]) for prev in used_tests)
   score=row_pen+rate_pen+overlap_pen*2+new_pen*2
   if score<best_score:best_score=score;best=assign.copy()
  if best is None:raise RuntimeError('No valid candidate')
  mapping=dict(zip(keys,best));new=df['_cluster'].map(mapping).to_numpy();folder=a.out/f'split_{iteration+2}';folder.mkdir(exist_ok=True)
  for j,name in enumerate(['train','valid','test']):
   subset=df.loc[new==j].drop(columns=['_original_split','_cluster']);subset.to_csv(folder/f'{name}.csv',index=False)
  pd.DataFrame({'protein_cluster':keys,'split':[['train','valid','test'][int(v)] for v in best]}).to_csv(folder/'cluster_assignment.csv',index=False)
  sets=[{keys[i] for i in np.flatnonzero(best==j)} for j in range(3)]
  assert not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
  test_keys=sets[2];used_tests.append(test_keys)
  print(f'[split {iteration+2}] score={best_score:.5f} rows={[int((new==j).sum()) for j in range(3)]} rates={[round(float((pd.to_numeric(df.loc[new==j,"y"])>=12.1).mean()),4) for j in range(3)]} original_test_cluster_overlap={len(test_keys & original_test)} previous_test_overlaps={[len(test_keys & prev) for prev in used_tests[:-1]]}',flush=True)
 save_analysis(df,a.out)
 print('[done] output:',a.out.resolve())
if __name__=='__main__':main()
