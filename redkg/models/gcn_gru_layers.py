from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from redkg.config import Config
from redkg.evaluator import Evaluator
from redkg.models.graph_convolution import GraphConvolution
from redkg.models.kge import KGEModel
from redkg.utils import pickle_load
from torch import Tensor


class GCNGRU(nn.Module):
    """GCN + GRU layer"""

    def __init__(self, config: Config, nfeat: int, entity_vocab: Dict, relation_vocab: Dict) -> None:
        super().__init__()
        # self.kge_model.entity_embedding = torch.nn.Embedding(entity_max, nfeat)
        # uniform_range = 6 / np.sqrt(nfeat)
        # self.kge_model.entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)

        # self.relation_emb = torch.nn.Embedding(relation_max, nfeat)
        # uniform_range = 6 / np.sqrt(nfeat)
        # self.relation_emb.weight.data.uniform_(-uniform_range, uniform_range)

        entity_max = max(entity_vocab.values())
        relation_max = max(relation_vocab.values())
        self.n_hop_kg = pickle_load(f"{config.preprocess_results_dir}/n_hop_kg.pkl")

        self.kge_model = KGEModel(
            model_name="TransE",
            nentity=entity_max,
            nrelation=relation_max,
            hidden_dim=nfeat,
            gamma=12.0,
            double_entity_embedding=True,
            double_relation_embedding=True,
            evaluator=Evaluator(),
        )
        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)

        self.h = torch.randn(2, 1, 20)
        self.gru = nn.GRU(50, 20, 2)

        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def get_n_hop(self, entity_id: int) -> Dict[int, List[int]]:

        return self.n_hop_kg[entity_id]

    def distance(self, triplets: Tensor) -> Tensor:
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (
            self.kge_model.entity_embedding.weight[heads]
            + self.relation_emb.weight[relations]
            - self.kge_model.entity_embedding.weight[tails]
        ).norm(p=1, dim=1)

    def forward_gcn(self, x: Tensor) -> Tensor:
        # GCN
        out = F.relu(self.gc1(self.kge_model.entity_embedding.weight))
        out = self.gc2(out)
        out = F.log_softmax(out, dim=1)[x]
        return out

    def forward(self, x: Tensor) -> Tensor:
        # GCN
        out = F.relu(self.gc1(self.kge_model.entity_embedding.weight))
        out = self.gc2(out)
        out = F.log_softmax(out, dim=1)[x].reshape(1, 1, -1)

        # GRU
        out, self.h = self.gru(out, self.h)
        out = out.reshape(-1)
        return out


class GRU(nn.Module):
    def __init__(self, config: Config, nfeat: int, entity_vocab: Dict, relation_vocab: Dict) -> None:
        super(GRU, self).__init__()

        entity_max = max(entity_vocab.values())
        relation_max = max(relation_vocab.values())
        self.n_hop_kg = pickle_load(f"{config.preprocess_results_dir}/n_hop_kg.pkl")

        self.kge_model.entity_embedding = torch.nn.Embedding(entity_max, nfeat)
        uniform_range = 6 / np.sqrt(nfeat)
        self.kge_model.entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)

        self.relation_emb = torch.nn.Embedding(relation_max, nfeat)
        uniform_range = 6 / np.sqrt(nfeat)
        self.relation_emb.weight.data.uniform_(-uniform_range, uniform_range)

        self.h = torch.randn(2, 1, 20)
        self.gru = nn.GRU(50, 20, 2)

        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def get_n_hop(self, entity_id: int) -> Dict[int, List[int]]:
        return self.n_hop_kg[entity_id]

    def distance(self, triplets: Tensor) -> Tensor:
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (
            self.kge_model.entity_embedding.weight[heads]
            + self.relation_emb.weight[relations]
            - self.kge_model.entity_embedding.weight[tails]
        ).norm(p=1, dim=1)

    def transe_forward(self, pos_triplet: Tensor, neg_triplet: Tensor) -> Tensor:
        # -1 to avoid nan for OOV vector
        self.kge_model.entity_embedding.weight.data[:-1, :].div_(
            self.kge_model.entity_embedding.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True)
        )

        pos_distance = self.distance(pos_triplet)
        neg_distance = self.distance(neg_triplet)

        target = torch.tensor([-1], dtype=torch.long)

        return self.criterion(pos_distance, neg_distance, target)

    def forward_gcn(self, x: Tensor) -> Tensor:
        # GCN
        out = F.relu(self.gc1(self.kge_model.entity_embedding.weight))
        out = self.gc2(out)
        out = F.log_softmax(out, dim=1)[x]
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = self.kge_model.entity_embedding.weight[x]
        x = x.reshape(1, 1, -1)

        # GRU
        out, self.h = self.gru(x, self.h)
        out = out.reshape(-1)
        return out
