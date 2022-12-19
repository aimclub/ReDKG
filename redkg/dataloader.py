import random
from typing import Dict, List, Optional, Tuple, Type

import torch
from redkg.utils import *
from torch.utils.data import DataLoader, Dataset


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

                triples["head"].append(local_head)
                triples["relation"].append(local_relation)
                triples["tail"].append(local_tail)
    return triples, entity_vocab, item_vocab, relation_vocab


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, count):
        self.triples = triples
        self.len = len(triples["head"])
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count
        # self.true_head = true_head
        # self.true_tail = true_tail

    def _gen_negative(self, head, relation, tail):
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        return torch.randint(0, self.nentity, (self.negative_sample_size,)), subsampling_weight

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples["head"][idx], self.triples["relation"][idx], self.triples["tail"][idx]
        positive_sample = [head, relation, tail]

        negative_sample, subsampling_weight = self._gen_negative(head, relation, tail)
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

    def __getitem__(self, idx):
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
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
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
