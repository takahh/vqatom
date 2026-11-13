"""Fit the bondpath codebook on the same first 192 sorted reference shards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantize.bondpath_features import FeatureConfig,graph_features,intrinsic_keys,l2_normalize_rows


def torch_sparse(x,device):
    x=x.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(x.indptr,dtype=torch.int64,device=device),
        torch.as_tensor(x.indices,dtype=torch.int64,device=device),
        torch.as_tensor(x.data,dtype=torch.float32,device=device),size=x.shape,device=device)


def fit_group(x,k,seed,iters,device,batch_size=2048,normalization="l2"):
    """Spherical Lloyd k-means by default; no dimensional projection."""
    if normalization not in ("l2", "none"):raise ValueError("Unknown normalization")
    if normalization=="l2":x=l2_normalize_rows(x)
    x=x.astype(np.float32).tocsr()
    n,d=x.shape
    if not n:raise ValueError('Empty group')
    rng=np.random.default_rng(seed)
    # Initialize from distinct sampled vectors, not from all possible feature columns.
    pool=rng.choice(n,min(n,max(10000,4*k)),replace=False)
    sampled=x[pool].toarray()
    unique=np.unique(sampled,axis=0)
    effective=min(k,len(unique))
    if effective==1:
        center=np.asarray(x.mean(axis=0),dtype=np.float32).reshape(1,-1)
        if normalization=="l2":center/=np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-12)
        return np.repeat(center,k,axis=0),dict(points=n,effective_centers=1,inertia=None)
    # sklearn's kmeans++ is only an initializer; the full reference group is used
    # in every Lloyd iteration below.
    from sklearn.cluster import kmeans_plusplus
    init,_=kmeans_plusplus(unique,n_clusters=effective,random_state=seed,n_local_trials=2)
    centers=torch.as_tensor(init,dtype=torch.float32,device=device)
    previous=None;inertia=None
    for iteration in range(iters):
        sums=torch.zeros_like(centers);counts=torch.zeros(effective,dtype=torch.float32,device=device)
        center_norm=centers.square().sum(1);loss=0.;changed=0
        all_labels=[]
        for start in range(0,n,batch_size):
            block=x[start:start+batch_size].tocsr();xt=torch_sparse(block,device)
            distance=torch.as_tensor(np.asarray(block.multiply(block).sum(1)).ravel(),device=device)[:,None]+center_norm[None,:]-2*torch.sparse.mm(xt,centers.T)
            labels=distance.argmin(1);loss+=float(distance.gather(1,labels[:,None]).clamp_min(0).sum())
            rows=torch.as_tensor(np.repeat(np.arange(len(labels)),np.diff(block.indptr)),device=device)
            cols=torch.as_tensor(block.indices.astype(np.int64),device=device)
            values=torch.as_tensor(block.data,device=device)
            sums.view(-1).index_add_(0,labels[rows]*d+cols,values)
            counts+=torch.bincount(labels,minlength=effective).float()
            all_labels.append(labels.cpu().numpy())
        assigned=np.concatenate(all_labels)
        if previous is not None:changed=int(np.count_nonzero(assigned!=previous))
        nonempty=counts>0;centers[nonempty]=sums[nonempty]/counts[nonempty,None]
        if normalization=="l2":
            centers=torch.nn.functional.normalize(centers,p=2,dim=1)
        inertia=loss/n
        print(json.dumps(dict(event='lloyd_iteration',iteration=iteration+1,points=n,k=k,effective=effective,inertia=inertia,changed=changed)),flush=True)
        if previous is not None and changed==0:break
        previous=assigned
    result=centers.cpu().numpy()
    if effective<k:result=np.concatenate([result,result[np.arange(k-effective)%effective]],axis=0)
    return result,dict(points=n,effective_centers=effective,inertia=inertia)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--layout',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--codebook-size',type=int,choices=(1000,4000,5000,6000,10000),default=5000)
    p.add_argument('--shards',type=int,default=192);p.add_argument('--iters',type=int,default=20)
    p.add_argument('--correct-hydrogens',action='store_true',help='Repair column 6 from original smiles_<shard>.txt sidecars')
    p.add_argument('--rebuild-layout',action='store_true',help='Reallocate the fixed total budget over corrected H groups')
    p.add_argument('--reference-manifest',type=Path,help='Verify the historical reference shard hashes')
    p.add_argument('--normalization',choices=('l2','none'),default='l2')
    p.add_argument('--hop-layout',choices=('separate','integrated'),default='separate')
    p.add_argument('--omit-joint-path-counts',action='store_true',
        help='Retain typed hop aggregates and omit the joint bond/element path counts')
    p.add_argument('--seed',type=int,default=0);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    layout=json.loads(args.layout.read_text())
    config=FeatureConfig(include_joint_path_counts=not args.omit_joint_path_counts,
        hop_layout=args.hop_layout)
    mats=[];keys=[];source=[]
    expected=json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
    files=sorted(args.data.glob('adj_*.npy'))[:args.shards]
    if len(files)!=args.shards:raise ValueError('Reference shard count differs')
    if expected and [p.name for p in files]!=[r['adj'] for r in expected]:raise ValueError('Reference shard selection differs')
    if not files:raise ValueError('Reference corpus missing')
    for fi,path in enumerate(files):
        attr_path=path.with_name(path.name.replace('adj_','attr_',1))
        a=np.load(path,mmap_mode='r').reshape(-1,100,100);f=np.load(attr_path,mmap_mode='r').reshape(-1,100,79)
        if len(a)!=len(f):raise ValueError('Reference shard mismatch')
        record=dict(adj=path.name,attr=attr_path.name,adj_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),attr_sha256=hashlib.sha256(attr_path.read_bytes()).hexdigest())
        if expected and record!=expected[fi]:raise ValueError('Historical reference SHA256 differs: '+path.name)
        smiles=None
        if args.correct_hydrogens:
            sidecar=path.with_name(path.name.replace('adj_','smiles_',1)).with_suffix('.txt')
            smiles=sidecar.read_text().splitlines()
            if len(smiles)!=len(a):raise ValueError('Reference SMILES count mismatch')
            record.update(smiles=sidecar.name,smiles_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
        for mi,(adj,attrs) in enumerate(zip(a,f)):
            mask=np.abs(attrs).sum(1)>0;n=int(mask.sum())
            if n==0:continue
            if not np.array_equal(np.flatnonzero(mask),np.arange(n)):raise ValueError('Noncontiguous atom packing')
            real=np.asarray(attrs[:n],dtype=np.float32);bond=np.asarray(adj[:n,:n],dtype=np.float32).copy();np.fill_diagonal(bond,0)
            if smiles is not None:
                from quantize.bondpath_reference import repair_hydrogens
                real=repair_hydrogens(real,smiles[mi])
            mats.append(graph_features(bond,real,config));keys.extend(intrinsic_keys(real))
        source.append(record)
        print(json.dumps(dict(event='reference_shard',index=fi+1,total=len(files),atoms=len(keys))),flush=True)
    x=sparse.vstack(mats,format='csr');del mats
    if not np.any(x[:,132].data):
        raise ValueError('Reference hydrogen counts are all zero; regenerate shards with total_h_v2 and rebuild the intrinsic-group layout')
    if args.normalization=='l2':x=l2_normalize_rows(x)
    active=np.unique(x.indices);x=x[:,active].tocsr()
    print('ACTIVE ZERO-COLUMN ELIMINATION',len(active),'OF',config.feature_dim,'ATOMS',x.shape[0],flush=True)
    key_array=np.asarray(keys)
    groups={k:np.flatnonzero(key_array==k) for k in set(keys)}
    if args.rebuild_layout:
        from quantize.bondpath_reference import rebuild_layout
        layout=rebuild_layout(keys,layout,total={4000: 4021, 6000: 6021}.get(args.codebook_size))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.with_suffix('.layout.json').write_text(json.dumps(layout,indent=2))
    missing=set(groups)-set(layout['centers_per_group'])
    # Fixed intrinsic group budgets are copied from the old CB5000 checkpoint.
    # A previously unseen reference group is an error rather than silent merging.
    if missing:raise ValueError('Reference groups absent from baseline: '+repr(sorted(missing)))
    centers={};stats={}
    for number,(key,k) in enumerate(sorted(layout['centers_per_group'].items())):
        indices=groups.get(key)
        if indices is None or not len(indices):
            raise ValueError('Baseline group absent from the same reference shards: '+key)
        print('FITTING GROUP',number+1,key,'N',len(indices),'K',k,flush=True)
        centers[key],stats[key]=fit_group(x[indices],k,args.seed,args.iters,args.device,
            normalization=args.normalization)
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    metadata=layout['global_id_meta']
    if sum(len(c) for c in centers.values())!=metadata['global_vocab_size']:raise ValueError('Global vocabulary mismatch')
    from quantize import bondpath_reference as repair_module
    checkpoint=dict(config=dict(feature_config=config.to_dict(),codebook_size=args.codebook_size,seed=args.seed,kmeans_iters=args.iters,
        semantic_hop_mode=config.mode,projection='none',reference_shards=len(files),normalization=args.normalization,hydrogen_count_version='total_h_v2',hydrogen_reference_parser='strict_then_permissive_remove_hs_v1',
        hydrogen_reference_parser_fallbacks=repair_module.REPAIR_FALLBACK_COUNT,
        hydrogen_reference_source_sha256=hashlib.sha256(Path(repair_module.__file__).read_bytes()).hexdigest(),active_columns_preserve_full_distance=True),
        active_columns=torch.from_numpy(active.astype(np.int64)),
        centers={k:torch.from_numpy(v) for k,v in centers.items()},global_id_meta=metadata,
        reference_checkpoint_sha256=layout['reference_checkpoint_sha256'],reference_sources=source,fit_stats=stats,
        group_layout=layout,hydrogen_repaired_from_sidecars=args.correct_hydrogens)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix('.partial');torch.save(checkpoint,temporary);temporary.replace(args.out)
    print('CHECKPOINT READY',args.out,'VOCAB',metadata['global_vocab_size'],flush=True)


if __name__=='__main__':main()
