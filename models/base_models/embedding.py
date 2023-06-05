import torch
import torch.nn as nn

from models.base_models.ivpvae_components import CompressorTanh, DilatorTanh
from models.base_models.graph_models import MTGNN, GCNConv, GraphConstructor, GCN2, GraphConstructorGDN, GraphConstructorGDN3, GraphConstructorGDN4, GraphConstructorPang, GraphConstructorGDN2

import utils


# class Embedding_MLP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim * 2, 200),
#             nn.Tanh(),
#             nn.Linear(200, out_dim)
#         )

#     def forward(self, truth, mask):
#         x = torch.cat((truth, mask), -1)
#         assert (not torch.isnan(x).any())
#         return self.layers(x)


# class Embedding_MLP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim * 2, out_dim),
#             nn.Tanh()
#         )

#     def forward(self, truth, mask):
#         x = torch.cat((truth, mask), -1)
#         assert (not torch.isnan(x).any())
#         return self.layers(x)


# class Embedding_MLP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim * 2, out_dim),
#             nn.ReLU()
#         )
#         utils.init_network_weights(self.layers, method='kaiming_uniform_')

#     def forward(self, truth, mask):
#         x = torch.cat((truth, mask), -1)
#         assert (not torch.isnan(x).any())
#         return self.layers(x)


class Embedding_GNN(nn.Module):

    def __init__(self, in_dim, out_dim, args):
        super().__init__()

        self.num_nodes = in_dim

        self.gnn_dim = args.gnn_dim

        self.gcn_true = True

        self.dilator = DilatorTanh(self.gnn_dim)

        self.compressor = CompressorTanh(self.gnn_dim)

        if args.gc_method == 'gdn':
            self.gc = GraphConstructorGDN(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'gc':
            self.gc = GraphConstructor(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'pang':
            self.gc = GraphConstructorPang(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'gdn2':
            self.gc = GraphConstructorGDN2(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'gdn3':
            self.gc = GraphConstructorGDN3(
                nnodes=in_dim, cosine=args.subgraph_cosine, dim=self.gnn_dim)
        elif args.gc_method == 'gdn4':
            self.gc = GraphConstructorGDN4(
                nnodes=in_dim, cosine=args.subgraph_cosine, dim=self.gnn_dim)
        else:
            raise ValueError('Unknown GC method')

        # self.idx = torch.arange(self.num_nodes)
        self.register_buffer('idx', torch.arange(
            self.num_nodes))

        if args.gcn == 'gcn':
            self.gcn = GCNConv(self.gnn_dim, self.gnn_dim)
        elif args.gcn == 'gcn2':
            self.gcn = GCN2(self.gnn_dim, self.gnn_dim)
        else:
            raise ValueError('Unknown GCN method')

        self.emb_with_mlp = args.emb_with_mlp
        if self.emb_with_mlp:
            self.latent_mapper = Embedding_MLP(in_dim, out_dim)
        else:
            self.latent_mapper = nn.Sequential(nn.Linear(in_dim, out_dim), )
            utils.init_network_weights(self.latent_mapper)
        self.emb_with_mask = args.emb_with_mask

    def forward(self, truth, mask):
        '''
          Forward pass
        '''
        adp = self.gc(self.idx)

        x = self.dilator(truth)
        # mask_padded = mask.unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])

        x = self.gcn(x, adp.nonzero().t().contiguous())

        # x = x * mask_padded + x_tmp * (1 - mask_padded)
        # x = self.compressor(x)
        x = self.compressor(x)

        if self.emb_with_mask:
            x = truth * mask + x * (1 - mask)

        if self.emb_with_mlp:
            x = self.latent_mapper(x, mask)
        else:
            x = self.latent_mapper(x)

        return x


class Embedding_GNN(nn.Module):

    def __init__(self, in_dim, out_dim, args):
        super().__init__()

        self.num_nodes = in_dim

        self.gnn_dim = args.gnn_dim

        self.gcn_true = True

        self.dilator = DilatorTanh(self.gnn_dim)

        self.compressor = CompressorTanh(self.gnn_dim)

        if args.gc_method == 'gdn':
            self.gc = GraphConstructorGDN(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'gc':
            self.gc = GraphConstructor(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'pang':
            self.gc = GraphConstructorPang(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'gdn2':
            self.gc = GraphConstructorGDN2(
                nnodes=in_dim, k=args.subgraph_size, dim=self.gnn_dim)
        elif args.gc_method == 'gdn3':
            self.gc = GraphConstructorGDN3(
                nnodes=in_dim, cosine=args.subgraph_cosine, dim=self.gnn_dim)
        elif args.gc_method == 'gdn4':
            self.gc = GraphConstructorGDN4(
                nnodes=in_dim, cosine=args.subgraph_cosine, dim=self.gnn_dim)
        else:
            raise ValueError('Unknown GC method')

        # self.idx = torch.arange(self.num_nodes)
        self.register_buffer('idx', torch.arange(
            self.num_nodes))

        if args.gcn == 'gcn':
            self.gcn = GCNConv(self.gnn_dim, self.gnn_dim)
        elif args.gcn == 'gcn2':
            self.gcn = GCN2(self.gnn_dim, self.gnn_dim)
        elif args.gcn == 'mtgnn':
            self.gcn = MTGNN(self.gnn_dim, self.gnn_dim)
        else:
            raise ValueError('Unknown GCN method')

        self.emb_with_mlp = args.emb_with_mlp
        if self.emb_with_mlp:
            self.latent_mapper = Embedding_MLP(in_dim, out_dim)
        else:
            self.latent_mapper = nn.Sequential(nn.Linear(in_dim, out_dim), )
            utils.init_network_weights(self.latent_mapper)
        self.emb_with_mask = args.emb_with_mask

    def forward(self, truth, mask):
        '''
          Forward pass
        '''
        adp = self.gc(self.idx)

        x = self.dilator(truth)
        # mask_padded = mask.unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])

        x = self.gcn(x, adp)

        # x = x * mask_padded + x_tmp * (1 - mask_padded)
        # x = self.compressor(x)
        x = self.compressor(x)

        if self.emb_with_mask:
            x = truth * mask + x * (1 - mask)

        if self.emb_with_mlp:
            x = self.latent_mapper(x, mask)
        else:
            x = self.latent_mapper(x)

        return x
