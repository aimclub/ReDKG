from __future__ import absolute_import, division, print_function

import logging
from collections import defaultdict
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from redkg.dataloader import BidirectionalOneShotIterator, TestDataset
from redkg.evaluator import Evaluator


class KGEModel(nn.Module):
    """An implementation of several knowledge graph embedding models.

    The following models were ...
    """

    def __init__(
        self,
        model_name: str,
        nentity: int,
        nrelation: int,
        hidden_dim: int,
        gamma: float,
        evaluator: Evaluator,
        double_entity_embedding: bool = False,
        double_relation_embedding: bool = False,
    ) -> None:
        super(KGEModel, self).__init__()
        """Initialize KGE model.

        :model_name: model name; only TransE, DistMult, ComplEx, RotatE models are currently unavailable
        :nentity: The entity
        :nrelation: The entity
        :gamma: The entity
        :evaluator: The entity
        :double_entity_embedding: The entity
        :double_relation_embedding: The entity

        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        """
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ["TransE", "DistMult", "ComplEx", "RotatE"]:
            raise ValueError("model %s not supported" % model_name)

        if model_name == "RotatE" and (not double_entity_embedding or double_relation_embedding):
            raise ValueError("RotatE should use --double_entity_embedding")

        if model_name == "ComplEx" and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError("ComplEx should use --double_entity_embedding and --double_relation_embedding")

        self.evaluator = evaluator

    def forward(self, sample: Tensor, mode: str = "single") -> Tensor:
        """Forward function that calculate the score of a batch of triples.

        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).

        :param sample: _description_
        :type sample: _type_
        :param mode: _description_, defaults to 'single'
        :type mode: str, optional
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return: _description_
        :rtype: _type_
        """
        if mode == "single":
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

        else:
            raise ValueError("mode %s not supported" % mode)

        model_func: Dict[str, Callable] = {
            "TransE": self.TransE,
            "DistMult": self.DistMult,
            "ComplEx": self.ComplEx,
            "RotatE": self.RotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError("model %s not supported" % self.model_name)

        return score

    def TransE(self, head: Tensor, relation: Tensor, tail: Tensor, mode: str) -> Tensor:
        """Apply TransE model

        :param head: (Tensor) heads of the triples
        :param relation: (Tensor) relations of the triples
        :param tail: (Tensor) heads of the triples
        :param mode: (str) mode of processing
        :returns: (Tensor) calculated score
        """
        if mode == "head-batch":
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head: Tensor, relation: Tensor, tail: Tensor, mode: str) -> Tensor:
        """Apply DistMult model

        :param head: (Tensor) heads of the triples
        :param relation: (Tensor) relations of the triples
        :param tail: (Tensor) heads of the triples
        :param mode: (str) mode of processing
        :returns: (Tensor) calculated score
        """
        if mode == "head-batch":
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head: Tensor, relation: Tensor, tail: Tensor, mode: str) -> Tensor:
        """Apply ComplEx model

        :param head: (Tensor) heads of the triples
        :param relation: (Tensor) relations of the triples
        :param tail: (Tensor) heads of the triples
        :param mode: (str) mode of processing
        :returns: (Tensor) calculated score
        """
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head: Tensor, relation: Tensor, tail: Tensor, mode: str) -> Tensor:
        """Apply RotatE model

        :param head: (Tensor) heads of the triples
        :param relation: (Tensor) relations of the triples
        :param tail: (Tensor) heads of the triples
        :param mode: (str) mode of processing
        :returns: (Tensor) calculated score
        """
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    @staticmethod
    def train_step(
        model: nn.Module, optimizer: Optimizer, train_iterator: BidirectionalOneShotIterator, model_parameters: Any
    ) -> Dict:
        """A single train step. Apply back-propation and return the loss

        :param model: model to train
        :param optimizer: _description_
        :param train_iterator: _description_
        :param model_parameters: _description_
        :return: Dict with steps logs
        """
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if model_parameters.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if model_parameters.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (
                F.softmax(negative_score * model_parameters.adversarial_temperature, dim=1).detach()
                * F.logsigmoid(-negative_score)
            ).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if model_parameters.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if model_parameters.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = model_parameters.regularization * (
                model.entity_embedding.norm(p=3) ** 3 + model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {"regularization": regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            "positive_sample_loss": positive_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item(),
        }

        return log

    @staticmethod
    def test_step(model: nn.Module, test_triples: Tensor, args: Any, random_sampling: bool = False) -> Dict[str, float]:
        """Evaluate the model on tests or valid datasets

        :param model: _description_
        :type model: _type_
        :param test_triples: _description_
        :type test_triples: _type_
        :param args: _description_
        :type args: _type_
        :param random_sampling: _description_, defaults to False
        :type random_sampling: bool, optional
        :return: _description_
        :rtype: _type_
        """
        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(test_triples, args, "head-batch", random_sampling),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(test_triples, args, "tail-batch", random_sampling),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    score = model((positive_sample, negative_sample), mode)

                    batch_results = model.evaluator.eval({"y_pred_pos": score[:, 0], "y_pred_neg": score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info("Evaluating the model... (%d/%d)" % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
