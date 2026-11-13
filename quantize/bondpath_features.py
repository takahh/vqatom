"""Unprojected, bond-conditioned chemical counts; no averaging or hashing.

Columns are fixed independently of the fitted corpus. Globally unused columns
may be omitted from stored centroids: their contribution to distance is the
same for every centroid, so this does not change nearest-centroid assignment.
"""
from dataclasses import asdict,dataclass
import re

import numpy as np
from scipy import sparse

MODE='bondpath_full_v1'


@dataclass(frozen=True)
class FeatureConfig:
    mode:str=MODE
    atomic_number_count:int=119
    hybridization_count:int=9
    ring_size_count:int=101
    hop1_weight:float=.4
    hop2_weight:float=.3
    include_bond_environment:bool=True
    include_joint_path_counts:bool=True
    hop_layout:str='separate'
    projection:str='none'
    aggregation:str='bond_conditioned_sum_and_joint_path_counts'

    @property
    def atom_dim(self):
        return self.atomic_number_count+1+1+self.hybridization_count+2+1+18+2+self.ring_size_count+1+(48 if self.include_bond_environment else 0)

    @property
    def offsets(self):
        d=self.atom_dim
        if self.hop_layout=='integrated':return dict(integrated=0,hop2_joint=16*d)
        return dict(center=0,hop1=d,hop2_terminal=5*d,
                    hop2_joint=21*d)

    @property
    def feature_dim(self):return (16 if self.hop_layout=='integrated' else 21)*self.atom_dim+16*self.atomic_number_count**2

    def to_dict(self):return dict(asdict(self),atom_dim=self.atom_dim,feature_dim=self.feature_dim)

    @classmethod
    def from_dict(cls,raw):
        fields=cls.__dataclass_fields__
        conf=cls(**{k:v for k,v in raw.items() if k in fields})
        if conf.mode!=MODE or conf.projection!='none':raise ValueError('Wrong feature mode')
        if conf.hop_layout not in ('separate','integrated'):raise ValueError('Unknown hop layout')
        for k in ('atom_dim','feature_dim'):
            if k in raw and raw[k]!=getattr(conf,k):raise ValueError('Feature schema dimension mismatch')
        return conf


def intrinsic_keys(attrs):
    attrs=np.asarray(attrs)
    return ['k_'+re.sub(r'[^0-9A-Za-z_]','_','_'.join(str(int(v)) for v in row[[0,2,3,4,5,6]])) for row in attrs]


def atom_features(attrs,config):
    x=np.asarray(attrs,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=79 or not np.isfinite(x).all():raise ValueError('Expected finite [N,79] atom features')
    n=len(x);out=np.zeros((n,config.atom_dim),dtype=np.float32);offset=0;rows=np.arange(n)
    def one_hot(values,size):
        nonlocal offset
        ids=values.astype(np.int64)
        if not np.array_equal(ids,values) or np.any(ids<0) or np.any(ids>=size):
            raise ValueError('Categorical feature outside configured schema')
        out[rows,offset+ids]=1;offset+=size
    def values(a):
        nonlocal offset
        a=np.asarray(a,dtype=np.float32).reshape(n,-1);out[:,offset:offset+a.shape[1]]=a;offset+=a.shape[1]
    one_hot(x[:,0],config.atomic_number_count)
    values(x[:,1]/6);values(x[:,2])  # preserve degree and charge magnitude, without clipping
    one_hot(x[:,3],config.hybridization_count)
    values(x[:,4:6]);values(x[:,6]/4);values(x[:,7:25]);values(x[:,25:27])
    one_hot(x[:,27],config.ring_size_count);values(x[:,28]/4)
    if config.include_bond_environment:values(x[:,31:79])
    if offset!=config.atom_dim:raise AssertionError('Atom schema mismatch')
    return out


def graph_features(adj,attrs,config=None):
    config=config or FeatureConfig();attrs=np.asarray(attrs,dtype=np.float32)
    if config.hop_layout not in ('separate','integrated'):raise ValueError('Unknown hop layout')
    adj=np.asarray(adj,dtype=np.float32);n=len(attrs)
    if adj.shape!=(n,n) or not np.isfinite(adj).all():raise ValueError('Bad adjacency')
    if not np.array_equal(adj,adj.T):raise ValueError('Asymmetric bond matrix')
    if np.any(np.diag(adj)!=0):raise ValueError('Self loops are not chemical bonds')
    if not np.isin(adj,[0,1,2,3,4]).all():raise ValueError('Unknown bond encoding')
    base=sparse.csr_matrix(atom_features(attrs,config));d=config.atom_dim;z=attrs[:,0].astype(np.int64)
    # Typed 1-hop blocks retain the complete neighbor atom descriptor.
    bonds=[sparse.csr_matrix((adj==b).astype(np.float32)) for b in (1,2,3,4)]
    blocks=[base]+[(a@base)*config.hop1_weight for a in bonds]
    joint_rows=[];joint_cols=[];joint_values=[]
    # Non-backtracking two-edge paths; atom order is never used as a feature.
    # Keep the intermediate/terminal element and both bonds as a JOINT tuple.
    terminal_blocks=[]
    for b1,a1 in enumerate(bonds):
        for b2,a2 in enumerate(bonds):
            counts=(a1@a2).tolil();counts.setdiag(0);counts=counts.tocsr();counts.eliminate_zeros()
            terminal_blocks.append((counts@base)*config.hop2_weight)
            if not config.include_joint_path_counts:continue
            for center in range(n):
                for middle in np.flatnonzero(adj[center]==b1+1):
                    for end in np.flatnonzero(adj[middle]==b2+1):
                        if end==center:continue
                        col=(((b1*4+b2)*config.atomic_number_count+z[middle])*config.atomic_number_count+z[end])
                        joint_rows.append(center);joint_cols.append(col);joint_values.append(config.hop2_weight)
    joint=sparse.coo_matrix((joint_values,(joint_rows,joint_cols)),shape=(n,16*config.atomic_number_count**2),dtype=np.float32).tocsr()
    if config.hop_layout=='integrated':
        # Embed each radius into the same ordered bond-pair channels. A one-hop
        # feature retains its first bond type and is uniform over the second.
        # 16 copies of h0/4 and 4 copies of h1/2 preserve each component's
        # squared norm. Cross-radius inner products are the intended change.
        terminal_blocks=[terminal_blocks[b1*4+b2]+blocks[0]*.25+blocks[b1+1]*.5
                         for b1 in range(4) for b2 in range(4)]
        components=terminal_blocks+[joint]
    else:
        components=blocks+terminal_blocks+[joint]
    result=sparse.hstack(components,format='csr',dtype=np.float32)
    result.sum_duplicates();result.eliminate_zeros()
    if result.shape!=(n,config.feature_dim) or not np.isfinite(result.data).all():raise ValueError('Invalid graph features')
    return result


def smiles_features(smiles,config=None,*,hydrogen_count_version="total_h_v2"):
    from rdkit import Chem
    from smiles_to_npy_discretize import smiles_to_graph_with_labels
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or not 1<=mol.GetNumAtoms()<=100:raise ValueError('Invalid SMILES or atom count')
    adj,attrs,_=smiles_to_graph_with_labels(smiles,idx=0,hydrogen_count_version=hydrogen_count_version)
    if len(attrs)!=mol.GetNumAtoms():raise ValueError('Atom order/count mismatch')
    return graph_features(adj,attrs,config),intrinsic_keys(attrs)


def l2_normalize_rows(x):
    """Normalize the FULL sparse descriptor before eliminating zero columns."""
    x=x.tocsr()
    norms=np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    inverse=np.zeros_like(norms)
    np.divide(1.,norms,out=inverse,where=norms>0)
    return x.multiply(inverse[:,None]).tocsr()
