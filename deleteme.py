import dgl
import torch
from dgl.dataloading import NeighborSampler

# Example graph
g = dgl.rand_graph(1000, 5000)  # Graph with 1000 nodes and 5000 edges

# Training node indices
train_idx = torch.arange(800)  # First 800 nodes as training data

# Neighbor sampler: Sample 10 neighbors for each node
sampler = NeighborSampler([10])

# Create a DataLoader
dataloader = dgl.dataloading.DataLoader(
    g,
    train_idx,
    sampler,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=4,
)

# Iterate through the DataLoader
for input_nodes, output_nodes, blocks in dataloader:
    print("Input Nodes:", input_nodes)
    print("Output Nodes:", output_nodes)
    print("Blocks:", blocks)
    break
