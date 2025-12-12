# src/training/models/graphsage.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    Node-ID embedding + GraphSAGE encoder + linear prediction head.

    Intended for minibatch training with DGL blocks:
      forward(blocks, input_nodes) -> logits for output nodes of last block.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        aggregator_type: str = "mean",
    ):
        super().__init__()
        assert num_layers >= 1

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_nodes, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SAGEConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    aggregator_type=aggregator_type,
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.layers:
            if hasattr(layer, "fc_self"):
                nn.init.xavier_uniform_(layer.fc_self.weight)
                if layer.fc_self.bias is not None:
                    nn.init.zeros_(layer.fc_self.bias)
            if hasattr(layer, "fc_neigh"):
                nn.init.xavier_uniform_(layer.fc_neigh.weight)
                if layer.fc_neigh.bias is not None:
                    nn.init.zeros_(layer.fc_neigh.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, blocks, input_nodes):
        """
        blocks: list[dgl.DGLBlock] from DGL's NeighborSampler
        input_nodes: 1D tensor of node IDs corresponding to blocks[0].srcdata
        """
        h = self.embedding(input_nodes)  # [num_src_nodes, hidden_dim]

        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)      # DGL will pick src/dst partition
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        logits = self.fc_out(h)          # [num_dst_nodes_of_last_block, out_dim]
        return logits
