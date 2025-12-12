# src/models/gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):
    """
    Node-ID embedding + multi-layer GCN + linear prediction head.

    Designed to work with DGL's neighbor sampling (block-based):
      forward(blocks, input_nodes) -> logits for output nodes of last block.

    Args
    ----
    num_nodes : int
        Total number of nodes in the graph.
    hidden_dim : int
        Dimension of the node embedding / hidden GCN layers.
    out_dim : int
        Number of output dimensions (e.g., num tasks = 4).
    num_layers : int
        Number of GraphConv layers.
    dropout : float
        Dropout rate applied after each non-final layer.
    norm : str
        Normalization mode for GraphConv ("none", "both", "right").
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm: str = "both",
    ):
        super().__init__()
        assert num_layers >= 1, "GCN must have at least one layer."

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Node ID embedding: each node_id -> hidden_dim vector
        self.embedding = nn.Embedding(num_nodes, hidden_dim)

        # Stack of GraphConv layers; all hidden_dim -> hidden_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    norm=norm,
                    activation=None,              # we apply activation manually
                    allow_zero_in_degree=True,    # <- critical for sampled blocks
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Final prediction head
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Embedding init
        nn.init.xavier_uniform_(self.embedding.weight)

        # GraphConv init
        for layer in self.layers:
            if hasattr(layer, "weight") and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Output layer init
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, blocks, input_nodes):
        """
        Forward pass for block-based mini-batch training.

        Parameters
        ----------
        blocks : list[dgl.DGLBlock]
            Blocks produced by DGL's NeighborSampler. blocks[0] contains
            the largest neighborhood; blocks[-1] corresponds to the last layer.
        input_nodes : torch.Tensor
            1D tensor of node IDs corresponding to blocks[0].srcdata.

        Returns
        -------
        logits : torch.Tensor
            Tensor of shape [num_output_nodes, out_dim], where num_output_nodes
            is the number of destination nodes in the last block.
        """
        # Node ID -> embedding
        h = self.embedding(input_nodes)  # [num_src_nodes_block0, hidden_dim]

        # Propagate through GCN layers
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        # h now corresponds to the dst nodes of the last block
        logits = self.fc_out(h)          # [num_dst_last_block, out_dim]
        return logits
