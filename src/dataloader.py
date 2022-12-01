import torch
from torch.utils.data import Dataset, DataLoader
import random

class TransE_dataset(Dataset): 
    def __init__(self, config, entity_vocab, relation_vocab):
        self.config = config
        self.positive_triples = []
        self.negative_triples = []
        with open(self.config.kg_path, encoding='utf8') as reader:
            for line in reader:
                head, relation, tail = line.strip().split('\t')
                local_head, local_tail = int(head), int(tail)
                if local_head in entity_vocab and local_tail in entity_vocab:
                    local_relation = relation_vocab[relation]

                    positive_triples = torch.stack((torch.tensor(entity_vocab[local_head]-1), torch.tensor(local_relation-1), torch.tensor(entity_vocab[local_tail]-1)), dim=0)

                    head_or_tail = torch.randint(high=2, size=(1,))
                    random_entities = random.choice(list(entity_vocab.keys()))
                    broken_heads = torch.where(head_or_tail == 1, random_entities, local_head).item()
                    broken_tails = torch.where(head_or_tail == 0, random_entities, local_tail).item()

                    negative_triples = torch.stack((torch.tensor(entity_vocab[broken_heads]-1), torch.tensor(local_relation-1), torch.tensor(entity_vocab[broken_tails]-1)), dim=0)
                    
                    self.positive_triples.append(positive_triples)
                    self.negative_triples.append(negative_triples)

    def __len__(self):
        return len(self.positive_triples)

    def __getitem__(self, index):
        return self.positive_triples[index], self.negative_triples[index]


def get_TransE_dataloader(config, entity_vocab, relation_vocab):
    dataset = TransE_dataset(config, entity_vocab, relation_vocab)
    return DataLoader(dataset, batch_size=128, shuffle=True)
