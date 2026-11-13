"""Read full, normalized bond-path atom descriptors from a shared sparse store."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch


class FullPrecisionSparseLinear(torch.nn.Linear):
    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            return torch.nn.functional.linear(features.float(), self.weight, self.bias)


class ContinuousFeatureStore:
    def __init__(self, metadata_path):
        path = Path(metadata_path).resolve()
        self.metadata = json.loads(path.read_text())
        matrix_path = path.parent / self.metadata['matrix']
        actual = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        if actual != self.metadata['matrix_sha256']:
            raise ValueError('Continuous feature matrix checksum differs')
        self.matrix = sparse.load_npz(matrix_path).tocsr()
        self.molecules = self.metadata['molecules']
        self.feature_dim = self.metadata['feature_dim']
        if self.matrix.shape != tuple(self.metadata['shape']):
            raise ValueError('Continuous feature matrix shape differs')
        if not np.isfinite(self.matrix.data).all():
            raise ValueError('Non-finite continuous features')
        norms = np.asarray(self.matrix.multiply(self.matrix).sum(axis=1)).ravel()
        if not np.allclose(norms, 1., atol=2e-6):
            raise ValueError('Continuous atom descriptors are not normalized')

    @classmethod
    @lru_cache(maxsize=4)
    def open(cls, metadata_path):
        return cls(metadata_path)

    def atom_count(self, key):
        entry = self.molecules[key]
        return entry['stop'] - entry['start']

    @lru_cache(maxsize=None)
    def tensor(self, key):
        entry = self.molecules[key]
        coo = self.matrix[entry['start']:entry['stop']].tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.copy())
        return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
