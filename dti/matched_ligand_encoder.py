"""Shared Transformer body for the restarted representation comparison."""
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch


class MatchedLigandEncoder(nn.Module):
    def __init__(self, *, vocab_size=None, base_vocab=None, pad_id=0, cls_id=1,
                 mask_id=None, atom_feat_dim=None, position_encoding=False, max_length=512):
        super().__init__()
        self.conf=dict(d_model=256,nhead=8,layers=6,dim_ff=1024,dropout=.1)
        self.pad_id=pad_id;self.cls_id=cls_id;self.mask_id=mask_id
        self.base_vocab=base_vocab;self.vocab_size=vocab_size
        self.atom_feat_dim=atom_feat_dim
        if atom_feat_dim is None:
            if vocab_size is None:raise ValueError('Token encoder needs a vocabulary')
            self.tok=nn.Embedding(vocab_size,256,padding_idx=pad_id)
        else:
            self.in_proj=nn.Linear(atom_feat_dim,256)
            self.cls=nn.Parameter(torch.zeros(1,1,256))
            nn.init.normal_(self.cls,std=.02)
        self.pos=nn.Embedding(max_length,256) if position_encoding else None
        layer=nn.TransformerEncoderLayer(d_model=256,nhead=8,dim_feedforward=1024,
            dropout=.1,activation='relu',batch_first=True,norm_first=False)
        self.enc=nn.TransformerEncoder(layer,num_layers=6)

    @property
    def d_model(self):return 256

    def forward(self,l_ids=None,edge_index=None,edge_batch=None,atom_feat=None,**kwargs):
        if self.atom_feat_dim is None:
            if l_ids is None:raise ValueError('Missing ligand IDs')
            x=self.tok(l_ids);padding=l_ids.eq(self.pad_id)
        else:
            if atom_feat is None or edge_batch is None:raise ValueError('Missing continuous atom features/batch')
            if atom_feat.ndim!=2 or atom_feat.shape[1]!=self.atom_feat_dim:raise ValueError('Wrong atom feature dimension')
            dense,valid=to_dense_batch(self.in_proj(atom_feat),edge_batch)
            x=torch.cat([self.cls.expand(dense.shape[0],-1,-1),dense],dim=1)
            padding=torch.cat([torch.zeros((valid.shape[0],1),device=valid.device,dtype=torch.bool),~valid],dim=1)
        if self.pos is not None:
            if x.shape[1]>self.pos.num_embeddings:raise ValueError('SMILES exceeds positional vocabulary')
            x=x+self.pos(torch.arange(x.shape[1],device=x.device)).unsqueeze(0)
        return self.enc(x,src_key_padding_mask=padding),padding
