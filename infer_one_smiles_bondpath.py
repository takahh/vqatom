"""Config-driven bondpath-full inference; one global ID for every atom."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from quantize.bondpath_features import FeatureConfig,MODE,smiles_features,l2_normalize_rows


class BondpathTokenizer:
    def __init__(self,path):
        self.path=Path(path);raw=self.path.read_bytes();self.checkpoint_sha256=hashlib.sha256(raw).hexdigest()
        ckpt=torch.load(self.path,map_location='cpu',weights_only=True)
        self.config=ckpt['config'];
        if self.config.get('normalization','none') not in ('none','l2'):raise ValueError('Unknown normalization')
        self.feature_config=FeatureConfig.from_dict(self.config['feature_config'])
        if self.config['semantic_hop_mode']!=MODE or self.config['codebook_size'] not in (1000,4000,5000,6000,10000):raise ValueError('Wrong mode/codebook')
        self.active=ckpt['active_columns'].numpy();self.meta=ckpt['global_id_meta'];self.global_vocab_size=int(self.meta['global_vocab_size'])
        self.centers={k:v.numpy() for k,v in ckpt['centers'].items()}
        self.center_norms={k:np.einsum('ij,ij->i',v,v) for k,v in self.centers.items()}
        if not np.array_equal(self.active,np.unique(self.active)) or np.any(self.active>=self.feature_config.feature_dim):raise ValueError('Bad active columns')
        for key,centers in self.centers.items():
            if centers.shape[1]!=len(self.active) or not np.isfinite(centers).all():raise ValueError('Bad centroid matrix')
        if sum(len(v) for v in self.centers.values())!=self.global_vocab_size:raise ValueError('Bad vocabulary size')

    @property
    def identity(self):return dict(checkpoint_sha256=self.checkpoint_sha256,mode=MODE,
        codebook_size=self.config['codebook_size'],global_vocab_size=self.global_vocab_size,feature_dim=self.feature_config.feature_dim,
        active_feature_dim=len(self.active),projection='none',normalization=self.config.get('normalization','none'),
        hydrogen_count_version=self.config.get('hydrogen_count_version','legacy_v1'))

    def encode(self,smiles):
        x,keys=smiles_features(smiles,self.feature_config,
            hydrogen_count_version=self.config.get('hydrogen_count_version','legacy_v1'))
        if self.config.get('normalization','none')=='l2':x=l2_normalize_rows(x)
        result=[];selected=x[:,self.active].tocsr()
        result=[None]*len(keys)
        for key in set(keys):
            if key not in self.centers:raise ValueError('Intrinsic atom group unseen in reference codebook: '+key)
            positions=[i for i,k in enumerate(keys) if k==key];centers=self.centers[key]
            distance=self.center_norms[key][None,:]-2*np.asarray(selected[positions]@centers.T)
            for i,local in zip(positions,distance.argmin(1)):
                result[i]=int(self.meta['global_offsets'][key])+int(local)
        if len(result)!=len(keys) or any(not 0<=v<self.global_vocab_size for v in result):raise ValueError('Atom/ID count or ID range mismatch')
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ckpt',required=True);p.add_argument('--smiles',required=True);a=p.parse_args()
    t=BondpathTokenizer(a.ckpt);print(json.dumps(dict(**t.identity,tokens=t.encode(a.smiles))))


if __name__=='__main__':main()
