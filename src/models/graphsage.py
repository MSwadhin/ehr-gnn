import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embed = nn.Embedding(num_nodes, hidden_dim)

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(hidden_dim, hidden_dim, "mean"))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, "mean"))
        self.layers.append(SAGEConv(hidden_dim, out_dim, "mean"))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, input_nodes, blocks):
        """
        DGL neighbor-sampling style forward.
        input_nodes: node IDs used to pull embeddings.
        blocks: list of DGLBlocks representing sampled neighborhoods.
        """
        h = self.embed(input_nodes)

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        return h
