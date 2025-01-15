import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from redkg.config import Config
from redkg.utils import pickle_dump

random.seed(14)
np.random.seed(14)


class DataPreprocessor:
    """Preprocess raw data to learning

    :params config: Config instance
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def _read_item2entity_file(self, item_vocab: Dict, entity_vocab: Dict) -> None:
        """_summary_

        :param item_vocab: Mapping from attribute file to item index in code
        :param entity_vocab: Mapping from attribute file to entity index in code
        """
        print(f"Logging Info - Reading item2entity file: {self._config.item2entity_path}")
        assert len(item_vocab) == 0 and len(entity_vocab) == 0
        with open(self._config.item2entity_path, encoding="utf8") as reader:
            for line in reader:
                item, entity = line.strip().split("\t")
                item = int(item)
                entity = int(entity)
                item_vocab[item] = len(item_vocab)
                entity_vocab[entity] = len(entity_vocab)

    def _read_attribute_file(
        self, attribute_path: str, user_vocab: Dict, item_vocab: Dict, entity_vocab: Dict
    ) -> Tuple[Dict, Dict, Dict]:
        print(f"Logging Info - Reading attribute file: {attribute_path}")
        assert len(user_vocab) == 0 and len(item_vocab) > 0
        user_attribute = defaultdict(list)

        # Save attribute datas into user_attribute dict
        with open(attribute_path, encoding="utf8") as reader:
            for idx, line in enumerate(reader):
                if idx == 0:  # Ignore first line
                    continue
                user, item, attribute, timestamp = line.strip().split(self._config.separator)[:4]
                user, item, attribute = int(user), int(item), float(attribute)
                if item in item_vocab:  # Ignore item not in KG
                    user_attribute[user].append((item_vocab[item], attribute, timestamp))

        # Remove user who has less interaction than 200 (minimum_interactions)
        # # of remained users has to be 16,525
        users2remove = []
        for k, v in user_attribute.items():
            if len(v) < self._config.minimum_interactions:
                users2remove.append(k)
        for user2remove in users2remove:
            del user_attribute[user2remove]

        # Split train : val : tests = 8 : 1 : 1
        remained_users = list(user_attribute.keys())
        total_users = len(remained_users)
        random.shuffle(remained_users)
        train_last_index = int(total_users * self._config.train_proportion)
        validation_last_index = int(total_users * (self._config.train_proportion + self._config.validation_proportion))
        train_users = remained_users[:train_last_index]
        val_users = remained_users[train_last_index:validation_last_index]

        # # of interactions has to be 6,711,013
        print("Logging Info - Converting attribute file...")
        train_attribute_dict: Dict[int, List[List]] = {}
        valid_attribute_dict: Dict[int, List[List]] = {}
        test_attribute_dict: Dict[int, List[List]] = {}
        unwatched_item_id_set = set(item_vocab.values())
        for user, rated_items_by_user in user_attribute.items():
            if user in train_users:
                target_dict = train_attribute_dict
            elif user in val_users:
                target_dict = valid_attribute_dict
            else:
                target_dict = test_attribute_dict

            user_vocab[user] = len(user_vocab)
            user_id = user_vocab[user]

            target_dict[user_id] = []
            for item_id, rate, timestamp in rated_items_by_user:
                if item_id in unwatched_item_id_set:
                    unwatched_item_id_set.remove(item_id)
                target_dict[user_id].append([item_id, rate, timestamp])

        # Remove unrated items
        # # of remained items has to be 16,426
        items2remove = []
        entities2remove = []
        for k, v in item_vocab.items():
            if v in unwatched_item_id_set:
                items2remove.append(k)
        for k, v in entity_vocab.items():
            if v in unwatched_item_id_set:
                entities2remove.append(k)

        for i in range(len(items2remove)):
            del item_vocab[items2remove[i]]
            del entity_vocab[entities2remove[i]]

        print(f"Logging Info - num of users: {len(user_vocab)}, num of items: {len(item_vocab)}")
        return train_attribute_dict, test_attribute_dict, valid_attribute_dict

    def _read_kg(
        self, entity_vocab: Dict, relation_vocab: Dict, user_vocab: Dict, item_vocab: Dict
    ) -> Tuple[Dict[int, Dict[int, List[int]]], NDArray]:
        print(f"Logging Info - Reading kg file: {self._config.kg_path}")

        kg = defaultdict(list)
        print("# user:", len(user_vocab), "# item:", len(item_vocab), "# entity:", len(entity_vocab))

        entity_freq = {}
        with open(self._config.kg_path, encoding="utf8") as reader:
            lines = reader.readlines()

        for line in lines:
            head, relation, tail = line.strip().split("\t")
            head, tail = int(head), int(tail)
            if tail not in entity_freq:
                entity_freq[tail] = 0
            else:
                entity_freq[tail] += 1

        max_entity_val = max(entity_vocab.values()) + 1
        while True:
            if len(entity_vocab) == 30000:
                break
            max_value = max(entity_freq.values())
            for k, v in entity_freq.items():
                if v == max_value:
                    entity_vocab[k] = max_entity_val
                    max_entity_val += 1
                    del entity_freq[k]
                    break

        for line in lines:
            head_str, relation, tail_str = line.strip().split("\t")
            head, tail = int(head_str), int(tail_str)
            if head not in entity_vocab or tail not in entity_vocab:
                continue
            else:
                if relation not in relation_vocab:
                    relation_vocab[relation] = len(relation_vocab)

            # Undirected graph
            kg[entity_vocab[head]].append(entity_vocab[tail])
            kg[entity_vocab[tail]].append(entity_vocab[head])

        print("# user:", len(user_vocab), "# item:", len(item_vocab), "# entity:", len(entity_vocab))
        max_entity = max(entity_vocab.values())
        adj_mat = np.zeros((max_entity, max_entity))

        for line in lines:
            head_str, relation, tail_str = line.strip().split("\t")
            head, tail = int(head_str), int(tail_str)
            if head in entity_vocab and tail in entity_vocab:
                adj_mat[entity_vocab[head] - 1][entity_vocab[tail] - 1] = 1
                adj_mat[entity_vocab[tail] - 1][entity_vocab[head] - 1] = 1

        n_hop_kg: Dict[int, Dict[int, List[int]]] = {}
        for entity in entity_vocab.values():
            n_hop_kg[entity] = {1: [], 2: []}
            n_hop_kg[entity][1] = kg[entity]
            for t in kg[entity]:
                n_hop_kg[entity][2].extend(kg[t])
            n_hop_kg[entity][2] = list(set(n_hop_kg[entity][2]) - set(n_hop_kg[entity][1]))

        print(
            f"Logging Info - num of entities: {len(entity_vocab)}, num of relations: {len(relation_vocab)}",
            "# adj:",
            adj_mat.sum() / 2,
        )
        return n_hop_kg, adj_mat

    def process_data(self) -> None:
        """Run preprocessing pipeline and save results to dir specified in config"""
        os.makedirs(self._config.preprocess_results_dir, exist_ok=True)

        # Sort attribute file based on user id and timestamp
        df = pd.read_csv(self._config.attribute_path, delimiter=self._config.separator)
        df = df.sort_values(by=["userId", "timestamp"], ascending=[True, True])
        sorted_attribute_path = f"{self._config.raw_data_dir}/{self._config.dataset_name}/sorted.csv"
        df.to_csv(sorted_attribute_path, index=False)

        user_vocab: Dict[int, Any] = {}
        item_vocab: Dict[int, Any] = {}
        entity_vocab: Dict[int, Any] = {}
        relation_vocab: Dict[int, Any] = {}

        self._read_item2entity_file(item_vocab, entity_vocab)
        train_data_dict, val_data_dict, test_data_dict = self._read_attribute_file(
            sorted_attribute_path, user_vocab, item_vocab, entity_vocab
        )
        pickle_dump(f"{self._config.preprocess_results_dir}/user_vocab.pkl", user_vocab)
        pickle_dump(f"{self._config.preprocess_results_dir}/item_vocab.pkl", item_vocab)
        pickle_dump(f"{self._config.preprocess_results_dir}/train_data_dict.pkl", train_data_dict)
        pickle_dump(f"{self._config.preprocess_results_dir}/val_data_dict.pkl", val_data_dict)
        pickle_dump(f"{self._config.preprocess_results_dir}/test_data_dict.pkl", test_data_dict)

        n_hop_kg, adj_mat = self._read_kg(entity_vocab, relation_vocab, user_vocab, item_vocab)
        pickle_dump(f"{self._config.preprocess_results_dir}/entity_vocab.pkl", entity_vocab)
        pickle_dump(f"{self._config.preprocess_results_dir}/relation_vocab.pkl", relation_vocab)
        pickle_dump(f"{self._config.preprocess_results_dir}/n_hop_kg.pkl", n_hop_kg)
        np.save(f"{self._config.preprocess_results_dir}/kg_adj_mat.npy", adj_mat)
