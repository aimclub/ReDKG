from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    """Dataset with training data"""

    def __init__(
        self,
        triples: Dict[str, Any],
        nentity: int,
        nrelation: int,
        negative_sample_size: int,
        mode: str,
        count: Dict,
        entity_dict: Optional[Dict[str, List[Any]]] = None,
        negative_mode: str = "full",
    ) -> None:
        self.len = len(triples["head"])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count
        self.entity_dict = entity_dict
        if negative_mode == "simple":
            self.negative_sample = self._gen_negative_s
        elif negative_mode == "full":
            self.negative_sample = self._gen_negative_f

    def __len__(self) -> int:
        return self.len

    def _gen_negative_s(
        self, head: int, relation: int, tail: int, head_type: str, tail_type: str
    ) -> Tuple[Tensor, Tensor]:
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        return torch.randint(0, self.nentity, (self.negative_sample_size,)), subsampling_weight

    def _gen_negative_f(
        self, head: int, relation: int, tail: int, head_type: str, tail_type: str
    ) -> Tuple[Tensor, Tensor]:
        subsampling_weight = self.count[(head, relation, head_type)] + self.count[(tail, -relation - 1, tail_type)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        if self.mode == "head-batch":
            negative_sample = torch.randint(
                self.entity_dict[head_type][0], self.entity_dict[head_type][1], (self.negative_sample_size,)  # type: ignore
            )
        elif self.mode == "tail-batch":
            negative_sample = torch.randint(
                self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], (self.negative_sample_size,)  # type: ignore
            )
        else:
            raise

        return negative_sample, subsampling_weight

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, str]:
        head, relation, tail = self.triples["head"][idx], self.triples["relation"][idx], self.triples["tail"][idx]
        if self.entity_dict:
            head_type, tail_type = self.triples["head_type"][idx], self.triples["tail_type"][idx]
            positive_sample = [head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]]
        else:
            positive_sample = [head, relation, tail]
            head_type, tail_type = None, None

        negative_sample, subsampling_weight = self.negative_sample(head, relation, tail, head_type, tail_type)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data: List[Any]) -> Tuple[Tensor, Tensor, Tensor, str]:
        """Collate

        :param data: data to collate
        :returns: (Tuple[Tensor, Tensor, Tensor, str]) positive_sample, negative_sample, subsample_weight, mode
        """
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode


class TestDataset(Dataset):
    """Dataset with test data"""

    def __init__(self, triples: Dict[str, Any], args: Any, mode: str, random_sampling: bool):
        self.len = len(triples["head"])
        self.triples = triples
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        head, relation, tail = self.triples["head"][idx], self.triples["relation"][idx], self.triples["tail"][idx]
        positive_sample = torch.LongTensor((head, relation, tail))

        if self.mode == "head-batch":
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([head]), torch.from_numpy(self.triples["head_neg"][idx])])
            else:
                negative_sample = torch.cat(
                    [torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.neg_size,))]
                )
        elif self.mode == "tail-batch":
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([tail]), torch.from_numpy(self.triples["tail_neg"][idx])])
            else:
                negative_sample = torch.cat(
                    [torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.neg_size,))]
                )
        else:
            raise ValueError(f"Not supported mode: {self.mode}")

        return positive_sample, negative_sample, self.mode

    @staticmethod
    def collate_fn(data: List[Any]) -> Tuple[Tensor, Tensor, str]:
        """Collate

        :param data: data to collate
        :returns: (Tuple[Tensor, Tensor, Tensor, str]) positive_sample, negative_sample, mode
        """
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode


class BidirectionalOneShotIterator(object):
    """Iterator over the data"""

    def __init__(self, dataloader_head: DataLoader, dataloader_tail: DataLoader) -> None:
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self) -> DataLoader:
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader: DataLoader) -> Generator:
        """Transform a PyTorch Dataloader into python iterator

        :param dataloader: Dataloader
        :returns: Generator of data
        """
        while True:
            for data in dataloader:
                yield data


def get_info(
    triples: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]],
    dataset: Optional[TrainDataset] = None,
    do_count: bool = False,
) -> Tuple[int, int, int, int, int, str]:
    """Get dataset info

    :param dataset: Dataset
    :param triples: (Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]) Dicts with triples splitted by train test and validation
    :returns: Tuple[number of entities, numbers of relations, volume train, volume validdation, volume_test, info_log]
    """
    if dataset is None:
        train, test, valid = triples
        nentity = len(pd.concat([train["head"], valid["head"], test["head"]]).unique()) + len(
            pd.concat([train["tail"], valid["tail"], test["tail"]]).unique()
        )
        nrelation = len(pd.concat([train["relation"], valid["relation"], test["relation"]]).unique())
        volume_train = round(len(train) / (len(train) + len(test) + len(valid)) * 100)
        volume_valid = round(len(valid) / (len(train) + len(test) + len(valid)) * 100)
        volume_test = round(len(test) / (len(train) + len(test) + len(valid)) * 100)
        info_log = f" NUMBER OF ENTITY: {nentity} \n NUMBER OF RELETION: {nrelation} \n TRAIN: {volume_train}% \n VALID: {volume_valid}% \n TEST: {volume_test}%"
        return nentity, nrelation, volume_train, volume_valid, volume_test, info_log

    entity_dict = dict()

    if do_count:
        nrelation = int(max(triples["relation"])) + 1  # type: ignore
        cur_idx = 0
        for key in dataset[0]["num_nodes_dict"]:  # type: ignore
            entity_dict[key] = (cur_idx, cur_idx + dataset[0]["num_nodes_dict"][key])  # type: ignore
            cur_idx += dataset[0]["num_nodes_dict"][key]  # type: ignore
        nentity = sum(dataset[0]["num_nodes_dict"].values())  # type: ignore
    else:
        nentity = dataset.nentity
        nrelation = dataset.nrelation

    count, true_head, true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)  # type: ignore
    for i in range(len(triples["head"])):  # type: ignore
        head, relation, tail = triples["head"][i], triples["relation"][i], triples["tail"][i]  # type: ignore
        if do_count:
            head_type, tail_type = triples["head_type"][i], triples["tail_type"][i]  # type: ignore
            count[(head, relation, head_type)] += 1
            count[(tail, -relation - 1, tail_type)] += 1
        else:
            count[(head, relation)] += 1  # type: ignore
            count[(tail, -relation - 1)] += 1  # type: ignore
        true_head[(relation, tail)].append(head)
        true_tail[(head, relation)].append(tail)

    info = {
        "nentity": nentity,
        "nrelation": nrelation,
        "count": count,
        "true_head": true_head,
        "true_tail": true_tail,
        "entity_dict": entity_dict,
    }

    return info  # type: ignore
