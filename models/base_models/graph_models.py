import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - \
            torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj*mask
        return adj


class GraphConstructorPang(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(GraphConstructorPang, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - \
            torch.mm(nodevec2, nodevec1.transpose(1, 0))

        adj = torch.tanh(self.alpha*a)
        adj_pang = torch.abs(adj + torch.rand_like(adj)*0.01)

        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.fill_(float('0'))

        s1, t1 = (adj_pang).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj*mask
        return adj


class GraphConstructorGDN(nn.Module):
    def __init__(self, nnodes, k, dim, static_feat=None):
        super(GraphConstructorGDN, self).__init__()
        self.embedding = nn.Embedding(nnodes, dim)
        self.topk = k

    def forward(self, idx):
        all_embeddings = self.embedding(idx)
        weights_arr = all_embeddings.detach().clone()
        weights = weights_arr.view(len(idx), -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(
            dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        _, idx_top = torch.topk(cos_ji_mat, self.topk, dim=-1)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.scatter_(1, idx_top, torch.ones(idx_top.size()).to(idx.device))
        adj = cos_ji_mat*mask
        return adj


class GraphConstructorGDN2(nn.Module):
    def __init__(self, nnodes, k, dim, static_feat=None):
        super(GraphConstructorGDN2, self).__init__()
        self.embedding = nn.Embedding(nnodes, dim)
        self.topk = k

    def forward(self, idx):
        all_embeddings = self.embedding(idx)
        weights_arr = all_embeddings.detach().clone()
        weights = weights_arr.view(len(idx), -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(
            dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        cos_ji_mat_pang = torch.abs(cos_ji_mat)
        _, idx_top = torch.topk(cos_ji_mat_pang, self.topk, dim=-1)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.scatter_(1, idx_top, torch.ones(idx_top.size()).to(idx.device))
        adj = cos_ji_mat*mask
        return adj


class GraphConstructorGDN3(nn.Module):
    def __init__(self, nnodes, cosine, dim, static_feat=None):
        super(GraphConstructorGDN3, self).__init__()
        self.embedding = nn.Embedding(nnodes, dim)
        self.cosine = cosine

    def forward(self, idx):
        all_embeddings = self.embedding(idx)
        weights_arr = all_embeddings.detach().clone()
        weights = weights_arr.view(len(idx), -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(
            dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        cos_ji_mat_pang = torch.abs(cos_ji_mat)

        # set mask value as 1 where values in cos_ji_mat_pang are greater than self.cosine
        mask = torch.gt(cos_ji_mat_pang, self.cosine).int()

        adj = cos_ji_mat*mask
        return adj


class GraphConstructorGDN4(nn.Module):
    def __init__(self, nnodes, cosine, dim, static_feat=None):
        super(GraphConstructorGDN4, self).__init__()
        self.embedding = nn.Embedding(nnodes, dim)
        self.cosine = cosine

    def forward(self, idx):
        all_embeddings = self.embedding(idx)
        weights_arr = all_embeddings.detach().clone()
        weights = weights_arr.view(len(idx), -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(
            dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))

        cos_ji_mat = cos_ji_mat / normed_mat
        # set mask value as 1 where values in cos_ji_mat_pang are greater than self.cosine
        mask = torch.gt(cos_ji_mat, self.cosine).int()
        adj = cos_ji_mat*mask

        return adj


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(-2), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        ho = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
        return ho


class MTGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gconv1 = mixprop(
            in_channels, out_channels, dropout=0.3, alpha=0.05)
        self.gconv2 = mixprop(
            in_channels, out_channels, dropout=0.3, alpha=0.05)

    def forward(self, x, adp):
        x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))
        return x
