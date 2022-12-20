from abc import ABC
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from redkg.config import Config
from redkg.evaluator import Evaluator
from redkg.models.graph_convolution import GraphConvolution
from redkg.models.kge import KGEModel
from redkg.utils import pickle_load


class AbstractLayer(nn.Module, ABC):
    """Abstract layer with KG and Basic GCN layers"""

    def __init__(self, config: Config, entity_vocab: Dict, relation_vocab: Dict, nfeat: int):
        super().__init__()
        self.config = config
        self.entity_max = max(entity_vocab.values())
        self.relation_max = max(relation_vocab.values())
        self.nfeat = nfeat
        self.n_hop_kg = pickle_load(f"{config.preprocess_results_dir}/n_hop_kg.pkl")

        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)

    def get_n_hop(self, entity_id: int) -> Dict[int, List[int]]:
        """Get n_hops from entity

        :param entity_id: id of entity
        :returns: Dict with N-hop neighbours
        """
        return self.n_hop_kg[entity_id]

    def distance(self, triplets: Tensor) -> Tensor:
        """Get distance between triples

        :param triplets: (Tensor) triplets to calculate
        :returns: (Tensor) distances
        """
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
        """Forward  GCN

        :param x: Tensor
        :returns: Tensor
        """
        out = F.relu(self.gc1(self.kge_model.entity_embedding.weight))
        out = self.gc2(out)
        out = F.log_softmax(out, dim=1)[x]
        return out


class GCNGRU(AbstractLayer):
    """GCN + GRU layer"""

    def __init__(
        self, config: Config, entity_vocab: Dict, relation_vocab: Dict, nfeat: int, model_name: str = "TransE"
    ) -> None:
        super().__init__(config, entity_vocab, relation_vocab, nfeat)
        self.kge_model = KGEModel(
            model_name=model_name,
            nentity=self.entity_max,
            nrelation=self.relation_max,
            hidden_dim=self.nfeat,
            gamma=12.0,
            double_entity_embedding=True,
            double_relation_embedding=True,
            evaluator=Evaluator(),
        )

        self.h = torch.randn(2, 1, 20)
        self.gru = nn.GRU(50, 20, 2)

        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward  GCN + GRU layers

        :param x: Tensor
        :returns: Tensor
        """
        # GCN
        out = F.relu(self.gc1(self.kge_model.entity_embedding.weight))
        out = self.gc2(out)
        out = F.log_softmax(out, dim=1)[x].reshape(1, 1, -1)

        # GRU
        out, self.h = self.gru(out, self.h)
        out = out.reshape(-1)
        return out


class GRU(AbstractLayer):
    """GRU layer"""

    def __init__(self, config: Config, nfeat: int, entity_vocab: Dict, relation_vocab: Dict) -> None:
        super().__init__(config, entity_vocab, relation_vocab, nfeat)
        self.kge_model.entity_embedding = torch.nn.Embedding(self.entity_max, nfeat)
        uniform_range = 6 / np.sqrt(nfeat)
        self.kge_model.entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)

        self.relation_emb = torch.nn.Embedding(self.relation_max, nfeat)
        uniform_range = 6 / np.sqrt(nfeat)
        self.relation_emb.weight.data.uniform_(-uniform_range, uniform_range)

        self.h = torch.randn(2, 1, 20)
        self.gru = nn.GRU(50, 20, 2)

        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def transe_forward(self, pos_triplet: Tensor, neg_triplet: Tensor) -> Tensor:
        """Forward with KGE model

        :param pos_triplet: (Tensor) positive triplets
        :param neg_triplet: (Tensor) negative triplets
        :returns: TransE scores
        """
        # -1 to avoid nan for OOV vector
        self.kge_model.entity_embedding.weight.data[:-1, :].div_(
            self.kge_model.entity_embedding.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True)
        )

        pos_distance = self.distance(pos_triplet)
        neg_distance = self.distance(neg_triplet)

        target = torch.tensor([-1], dtype=torch.long)

        return self.criterion(pos_distance, neg_distance, target)

    def forward(self, x: Tensor) -> Tensor:
        """Forward

        :param x: Tensor
        :returns: Tensor
        """
        x = self.kge_model.entity_embedding.weight[x]
        x = x.reshape(1, 1, -1)

        # GRU
        out, self.h = self.gru(x, self.h)
        out = out.reshape(-1)
        return out
