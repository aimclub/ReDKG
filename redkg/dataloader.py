import pickle
from collections import defaultdict
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def get_info(
    dataset: Dataset, 
    triples: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]],
    do_count: bool
) -> Tuple[int, int, int, int, int, str]:
    """Get dataset info

    :param dataset: Dataset
    :param triples: (Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]) Dicts with triples splitted by train test and validation
    :returns: Tuple[number of entities, numbers of relations, volume train, volume validdation, volume_test, info_log]
    """
    if not dataset:
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
        nrelation = int(max(triples["relation"])) + 1
        cur_idx = 0
        for key in dataset[0]["num_nodes_dict"]:
            entity_dict[key] = (cur_idx, cur_idx + dataset[0]["num_nodes_dict"][key])
            cur_idx += dataset[0]["num_nodes_dict"][key]
        nentity = sum(dataset[0]["num_nodes_dict"].values())
    else:
        nentity = dataset.nentity
        nrelation = dataset.nrelation

    count, true_head, true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    for i in range(len(triples["head"])):
        head, relation, tail = triples["head"][i], triples["relation"][i], triples["tail"][i]
        if do_count:
            head_type, tail_type = triples["head_type"][i], triples["tail_type"][i]
            count[(head, relation, head_type)] += 1
            count[(tail, -relation - 1, tail_type)] += 1
        else:
            count[(head, relation)] += 1
            count[(tail, -relation-1)] += 1
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

    return info


def get_TransE_dataloader(config, entity_vocab: Dict, relation_vocab: Dict):
    dataset = TrainDataset(config, entity_vocab, relation_vocab)
    return DataLoader(dataset, batch_size=128, shuffle=True)


def read_kg(path, kg_path):
    with open(path + "/entity_vocab.pkl", "rb") as f:
        entity_vocab = pickle.load(f)
    with open(path + "/item_vocab.pkl", "rb") as f:
        item_vocab = pickle.load(f)
    with open(path + "/relation_vocab.pkl", "rb") as f:
        relation_vocab = pickle.load(f)

    triples = {"head": [], "relation": [], "tail": []}
    with open(kg_path, encoding="utf8") as reader:
        for line in reader:
            head, relation, tail = line.strip().split("\t")
            local_head, local_tail = int(head), int(tail)
            if local_head in entity_vocab and local_tail in entity_vocab:
                local_relation = relation_vocab[relation]

                triples["head"].append(entity_vocab[local_head])
                triples["relation"].append(entity_vocab[local_relation])
                triples["tail"].append(entity_vocab[local_tail])
    return triples, entity_vocab, item_vocab, relation_vocab


class TrainDataset(Dataset):
    def __init__(
        self,
        triples,
        nentity,
        nrelation,
        negative_sample_size,
        mode,
        count,
        true_head=None,
        true_tail=None,
        entity_dict=None,
        negative_mode="full",
    ):
        self.len = len(triples["head"])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count
        self.true_head = true_head
        self.true_tail = true_tail
        self.entity_dict = entity_dict
        if negative_mode == "simple":
            self.negative_sample = self._gen_negative_s
        elif negative_mode == "full":
            self.negative_sample = self._gen_negative_f

    def __len__(self):
        return self.len

    def _gen_negative_s(self, head, relation, tail, head_type, tail_type):
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        return torch.randint(0, self.nentity, (self.negative_sample_size,)), subsampling_weight

    def _gen_negative_f(self, head, relation, tail, head_type, tail_type):
        subsampling_weight = self.count[(head, relation, head_type)] + self.count[(tail, -relation - 1, tail_type)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        if self.mode == "head-batch":
            negative_sample = torch.randint(
                self.entity_dict[head_type][0], self.entity_dict[head_type][1], (self.negative_sample_size,)
            )
        elif self.mode == "tail-batch":
            negative_sample = torch.randint(
                self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], (self.negative_sample_size,)
            )
        else:
            raise

        return negative_sample, subsampling_weight

    def __getitem__(self, idx):
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
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode


class TestDataset(Dataset):
    def __init__(self, triples, args, mode, random_sampling):
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

    def __getitem__(self, idx: int):
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
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader: DataLoader):
        """Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data

    # def __init__(self, config, entity_vocab: Dict, relation_vocab: Dict):
    #     self.config = config
    #     self.positive_triples = []
    #     self.negative_triples = []
    #     with open(self.config.kg_path, encoding='utf8') as reader:
    #         for line in reader:
    #             head, relation, tail = line.strip().split('\t')
    #             local_head, local_tail = int(head), int(tail)
    #             if local_head in entity_vocab and local_tail in entity_vocab:
    #                 local_relation = relation_vocab[relation]

    #                 positive_triples = torch.stack((torch.tensor(entity_vocab[local_head]-1), torch.tensor(local_relation-1), torch.tensor(entity_vocab[local_tail]-1)), dim=0)

    #                 head_or_tail = torch.randint(high=2, size=(1,))
    #                 random_entities = random.choice(list(entity_vocab.keys()))
    #                 broken_heads = torch.where(head_or_tail == 1, random_entities, local_head).item()
    #                 broken_tails = torch.where(head_or_tail == 0, random_entities, local_tail).item()

    #                 negative_triples = torch.stack((torch.tensor(entity_vocab[broken_heads]-1), torch.tensor(local_relation-1), torch.tensor(entity_vocab[broken_tails]-1)), dim=0)

    #                 self.positive_triples.append(positive_triples)
    #                 self.negative_triples.append(negative_triples)
