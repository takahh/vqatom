"""Versioned hydrogen repair and fixed-total intrinsic-group budget allocation."""
from collections import Counter
import numpy as np


REPAIR_FALLBACK_COUNT=0

def repair_hydrogens(attrs, smiles):
    """Change only column 6, checking SMILES atom order against reference atoms."""
    from rdkit import Chem
    global REPAIR_FALLBACK_COUNT
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        # Historical shards include hypervalent metal compounds rejected by
        # current sanitization. Preserve them and verify atom alignment below.
        mol=Chem.MolFromSmiles(smiles,sanitize=False)
        if mol is not None:
            mol=Chem.RemoveHs(mol,sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            REPAIR_FALLBACK_COUNT+=1
    if mol is None or mol.GetNumAtoms()!=len(attrs):
        raise ValueError('Reference SMILES atom count mismatch')
    if not np.array_equal(attrs[:,0], [a.GetAtomicNum() for a in mol.GetAtoms()]):
        raise ValueError('Reference SMILES atom order mismatch')
    out=np.array(attrs,copy=True)
    # Explicit neighboring H atoms and bracket/implicit H counts are both included.
    out[:,6]=[a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms()]
    return out


def rebuild_layout(keys, original, *, total=None):
    """Split legacy budgets over H states, retaining the exact total vocabulary.

    A minimum of one center/group is mandatory. Remaining slots are apportioned
    by legacy group budget times the new H state's frequency within that group,
    capped at group population. Deterministic largest-remainder allocation.
    """
    counts=Counter(keys)
    def legacy(k):return k.rsplit('_',1)[0]+'_0'
    parent_counts=Counter()
    for k,n in counts.items():parent_counts[legacy(k)]+=n
    old=original['centers_per_group'];total=sum(old.values()) if total is None else total
    if set(parent_counts)!=set(old):raise ValueError('Reference legacy group set differs')
    ordered=sorted(counts)
    weights=np.array([old[legacy(k)]*counts[k]/parent_counts[legacy(k)] for k in ordered])
    capacity=np.array([counts[k] for k in ordered],dtype=np.int64)
    allocated=np.ones(len(ordered),dtype=np.int64)
    if total<len(ordered) or total>int(capacity.sum()):raise ValueError('Infeasible fixed vocabulary budget')
    left=total-len(ordered)
    while left:
        available=allocated<capacity
        if not available.any():raise ValueError('Insufficient group capacity')
        w=weights*available;quota=left*w/w.sum()
        addition=np.minimum(np.floor(quota).astype(np.int64),capacity-allocated)
        if addition.sum():
            allocated+=addition;left-=int(addition.sum())
        else:
            # Stable ties follow the lexicographic intrinsic-key order.
            order=np.argsort(-quota,kind='stable')
            for i in order:
                if left and available[i]:allocated[i]+=1;left-=1
    budgets=dict(zip(ordered,map(int,allocated)));offsets={};cursor=0
    for k,n in budgets.items():offsets[k]=cursor;cursor+=n
    meta=dict(safe_keys=ordered,global_offsets=offsets,global_vocab_size=cursor,
              pad_id=cursor,mask_id=cursor+1,vocab_size=cursor+2,
              id2safe={i:k for i,k in enumerate(ordered)})
    return dict(centers_per_group=budgets,global_id_meta=meta,
        reference_checkpoint_sha256=original['reference_checkpoint_sha256'],
        allocation='minimum_one_then_legacy_budget_times_h_frequency_largest_remainder',
        original_centers_per_group=old)
