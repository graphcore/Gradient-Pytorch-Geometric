import poptorch
import torch
import torch.nn as nn
from torch_geometric.nn import Linear, SAGEConv

from loss import weighted_cross_entropy


# TODO: Use num layers?
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        assert num_layers > 1
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Model(torch.nn.Module):

    def __init__(self,
                 hetero_gnn,
                 embedding_size,
                 out_channels,
                 node_types,
                 num_nodes_per_type,
                 class_weight=None,
                 batch_size=None):
        super().__init__()
        self.hetero_gnn = hetero_gnn
        self.embedding = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes_per_type[node_type], embedding_size)
            for node_type in node_types
            if node_type != "transaction"
        })
        self.linear = Linear(-1, out_channels)
        self.node_types = node_types
        self.full_batch = (batch_size is None)
        self.batch_size = batch_size
        self.class_weight = class_weight

    def forward(self,
                x_dict,
                edge_index_dict,
                n_id_dict=None,
                target=None,
                mask=None):
        #with poptorch.Block(ipu_id=0):
        # TODO: Update this?
        for node_type in self.node_types:
            if node_type != "transaction":
                if self.full_batch:
                    x_dict[node_type] = self.embedding[node_type].weight
                else:
                    assert n_id_dict is not None, (
                        "If using a sampled batch, `n_id_dict` must"
                        " be provided.")
                    x_dict[node_type] = self.embedding[node_type](n_id_dict[node_type])

        #with poptorch.Block(ipu_id=1):
        x_dict = self.hetero_gnn(x_dict, edge_index_dict)
        out = self.linear(x_dict['transaction'])
        if self.training:
            if not self.full_batch:
                # TODO: Check this
                mask = (target * 0).bool()
                mask[:self.batch_size] = 1
            loss = weighted_cross_entropy(out, target, mask, self.class_weight)
            return out, loss
        return out
