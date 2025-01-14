import random
import sys

import numpy as np
import torch

from redkg.dataloader import get_info
from redkg.evaluator import Evaluator
from redkg.models.kge import KGEModel
from tests.utils import read_test_data

torch.manual_seed(0)

random.seed(0)

np.random.seed(0)

sys.path.append("../")

train, test, valid = read_test_data()


def test_data_info():
    nentity, nrelation, v_tr, v_val, v_test, info = get_info(triples=(train, test, valid))

    assert nentity == 20
    assert nrelation == 2
    assert v_tr == 71
    assert v_test == 14
    assert v_val == 14


def test_evaluator():
    evaluator = Evaluator()
    in_dict = {
        "y_pred_pos": torch.tensor([0.1]),
        "y_pred_neg": torch.tensor([[0.04, 0.11, 0.05, 0.03, 0.15, 0.07]]),
    }
    metrics = evaluator.eval(in_dict)
    assert round(sum(torch.cat(metrics).tolist()), 3) == 2.333


# score = model_func[self.model_name](head, relation, tail, mode)
def test_model():
    evaluator = Evaluator()
    kge_model = KGEModel(model_name="TransE", nentity=20, nrelation=2, hidden_dim=2, gamma=12,
                         evaluator=evaluator)
    assert list(kge_model.entity_embedding.shape) == [20, 2]
    assert list(kge_model.relation_embedding.shape) == [2, 2]

    head = torch.tensor([10])
    relation = torch.tensor([1])
    tail = torch.tensor([0])
    neg_head = torch.tensor([0, 2, 3])
    neg_tail = torch.tensor([12, 13, 14])

    pos_triple = torch.cat((head, relation, tail)).view(-1, 3)
    neg_tail_triple = torch.stack(
        (head.repeat(neg_tail.size()), relation.repeat(neg_tail.size()), neg_tail), dim=1)
    neg_head_triple = torch.stack(
        (neg_head, relation.repeat(neg_head.size()), tail.repeat(neg_head.size())), dim=1)
    triples = torch.cat((pos_triple, neg_tail_triple, neg_head_triple), dim=0)

    scores = kge_model(triples)

    positive_score = scores[: pos_triple.shape[0]]
    negative_tail_score = scores[
                          pos_triple.shape[0]: pos_triple.shape[0] + neg_tail_triple.shape[0]]
    negative_head_score = scores[
                          pos_triple.shape[0] + neg_tail_triple.shape[0]:
                          pos_triple.shape[0] + neg_tail_triple.shape[0] + neg_head_triple.shape[0]
                          ]

    assert (
               round(
                   sum(
                       [
                           x
                           for l_ in
                           positive_score.tolist() + negative_tail_score.tolist() + negative_head_score.tolist()
                           for x in l_
                       ]
                   ),
                   3,
               )
           ) == 28.427
