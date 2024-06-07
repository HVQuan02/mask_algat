import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.norm = nn.LayerNorm(out_feats)
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, adj):
        x = x.matmul(self.weight)
        x = adj.matmul(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class GraphModule(nn.Module):
    def __init__(self, num_layers, num_feats):
        super().__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

        layers = []
        for _ in range(num_layers):
            layers.append(GCNLayer(num_feats, num_feats))
        self.gcn = nn.ModuleList(layers)

    def forward(self, x, get_adj=False):
        qx = self.wq(x)
        kx = self.wk(x)
        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)

        for layer in self.gcn:
            x = layer(x, adj)

        x = x.mean(dim=-2)
        if get_adj is False:
            return x
        else:
            return x, adj


class ClassifierSimple(nn.Module):
    def __init__(self, num_feats, num_hid, num_class):
        super().__init__()
        self.fc1 = nn.Linear(num_feats, num_hid)
        self.fc2 = nn.Linear(num_hid, num_class)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class tokengraph_with_global_part_sharing(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.graph_omega = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2*num_feats, num_feats, num_class)

    def forward(self, feats, feats_global, get_adj=False):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        
        if get_adj is False:
            x = self.graph(feats)
            x = x.view(N, FR, NF)
            x = self.graph_omega(x)
            y = self.graph_omega(feats_global)
            x = torch.cat([x, y], dim=-1)
            x = self.cls(x)
            return x
        
        x, adjobj = self.graph(feats, get_adj)
        adjobj = adjobj.cpu()
        wids_objects = adjobj.numpy().sum(axis=1)
        x = x.view(N, FR, NF)

        x, adjframelocal = self.graph_omega(x, get_adj)
        adjframelocal = adjframelocal.cpu()
        wids_frame_local = adjframelocal.numpy().sum(axis=1)

        y, adjframeglobal = self.graph_omega(feats_global, get_adj)
        adjframeglobal = adjframeglobal.cpu()
        wids_frame_global = adjframeglobal.numpy().sum(axis=1)

        x = torch.cat([x, y], dim=-1)
        x = self.cls(x)

        return x, wids_objects, wids_frame_local, wids_frame_global


class MaskedGCN(nn.Module):
    def __init__(self, gcn_layers, num_feats, L, mask_percentage, is_global=False):
        super().__init__()
        self.is_global = is_global
        self.mask_percentage = mask_percentage
        self.masking_p = nn.Parameter(torch.randn(num_feats))  # Learnable masked vector p
        self.graph = GraphModule(gcn_layers, num_feats)
        self.fc = nn.Linear(num_feats, L)  # FC layer to transform F to L

    def forward(self, feats):
        if self.is_global:
            N, FR, NF = feats.shape

            # Masking features for each album
            for i in range(N):
                mask_indices = torch.randperm(FR)[:int(self.mask_percentage * FR)]
                feats[i, mask_indices] = self.masking_p

            x = self.graph(feats) # latent representation
            score_vector = self.fc(x)

        else:
            N, FR, B, NF = feats.shape
            feats = feats.view(N * FR, B, NF)
            
            # Masking features for each image
            for i in range(N * FR):
                mask_indices = torch.randperm(B)[:int(self.mask_percentage * B)]
                feats[i, mask_indices] = self.masking_p

            x = self.graph(feats)
            x = x.view(N, FR, -1)
            x = self.graph(x) # latent representation
            score_vector = self.fc(x)

        return score_vector
    

class cls_only(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.cls = ClassifierSimple(2*num_feats, num_feats, num_class)
    def forward(self, feats, feats_global):

        x = feats.mean(dim=-2)
        x = x.mean(dim=-2)
        y = feats_global.mean(dim=-2)
        x = torch.cat([x, y], dim=-1)
        x = self.cls(x)
        return x


class tokens_as_extra_Graph_mean(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.graph_omega = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(3*num_feats, num_feats, num_class)

    def forward(self, feats, feats_global):
        N, FR, B, NF = feats.shape

        feats = feats.view(N * FR, B, NF)
        x = self.graph_omega(feats)
        x = x.view(N, FR, NF)
        x = self.graph_omega(x)

        x_tokens = self.graph(feats)
        x_tokens = x_tokens.view(N, FR, NF)
        x_tokens = x_tokens.mean(dim=-2)

        y = self.graph_omega(feats_global)
        x = torch.cat([x, x_tokens, y], dim=-1)
        x = self.cls(x)
        return x


class tokenGraph_and_Graph(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.graph_omega3 = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        x_tokens = self.graph(feats)
        x_tokens = x_tokens.view(N, FR, NF)
        x_tokens = self.graph_omega3(x_tokens)
        x = self.cls(x_tokens)
        return x

class tokenGraph_and_Graph_shared(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        x_tokens = self.graph(feats)
        x_tokens = x_tokens.view(N, FR, NF)
        x_tokens = self.graph(x_tokens)
        x = self.cls(x_tokens)
        return x


class tokenGraph_and_mean(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        x_tokens = self.graph(feats)
        x_tokens = x_tokens.view(N, FR, NF)
        x_tokens = x_tokens.mean(dim=-2)
        x = self.cls(x_tokens)
        return x


class Graph_and_tokenGraph(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.graph_omega3 = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        x_tokens = self.graph_omega3(feats)
        x_tokens = x_tokens.view(N, FR, NF)
        x_tokens = self.graph(x_tokens)
        x = self.cls(x_tokens)
        return x


class mean_and_tokenGraph(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        x_tokens = feats.mean(dim=-2)
        x_tokens = x_tokens.view(N, FR, NF)
        x_tokens = self.graph(x_tokens)
        x = self.cls(x_tokens)
        return x