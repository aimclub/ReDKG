# type: ignore
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.functional import F
from torch.utils.data import DataLoader

from redkg.dataloader import BidirectionalOneShotIterator, TrainDataset
from redkg.env import Simulator
from redkg.models.basic_models import Net

# flake8: noqa


def train_gnn_model(model, optimizer, subgraph, positive_edges, negative_edges):
    # Обновление функции обучения
    model.train()
    optimizer.zero_grad()
    # subgraph.cuda()

    # Получаем эмбеддинги узлов
    node_embeddings = model(subgraph.x, subgraph.edge_index)

    # Подготовка меток и объединение положительных и отрицательных примеров
    labels = torch.cat([torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))], dim=0).to(subgraph.x.device)

    # Убедимся, что edges имеет правильный тип данных
    edges = torch.cat([torch.tensor(positive_edges), torch.tensor(negative_edges)], dim=0).to(subgraph.x.device).long()

    # Создаём эмбеддинги рёбер
    edge_embeddings = torch.cat([node_embeddings[edges[:, 0]], node_embeddings[edges[:, 1]]], dim=1)

    # Предсказание вероятности наличия связи
    predictions = torch.sigmoid(model.edge_predictor(edge_embeddings)).squeeze()

    # Вычисление потерь и обновление параметров модели
    loss = F.binary_cross_entropy(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_kge_model(kge_model, train_pars, info, train_triples, valid_triples, max_steps=10):
    """Trainin pipeline for model"""
    print("Training...")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, kge_model.parameters()), lr=train_pars.learning_rate)

    train_dataloader_head = DataLoader(
        TrainDataset(
            triples=train_triples,
            nentity=info["nentity"],
            nrelation=info["nrelation"],
            negative_sample_size=train_pars.negative_sample_size,
            mode="head-batch",
            count=info["count"],
            entity_dict=info["entity_dict"],
            negative_mode=train_pars["negative_mode"],
        ),
        batch_size=train_pars.train_batch_size,
        shuffle=True,
        num_workers=max(1, train_pars.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn,
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(
            triples=train_triples,
            nentity=info["nentity"],
            nrelation=info["nrelation"],
            negative_sample_size=train_pars.negative_sample_size,
            mode="tail-batch",
            count=info["count"],
            entity_dict=info["entity_dict"],
            negative_mode=train_pars["negative_mode"],
        ),
        batch_size=train_pars.train_batch_size,
        shuffle=True,
        num_workers=max(1, train_pars.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn,
    )
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    training_logs = []
    test_logs = []

    # Training Loop
    for step in range(max_steps):
        log = kge_model.train_step(kge_model, optimizer, train_iterator, train_pars)
        training_logs.append(log)

        if train_pars.do_test:
            metrics = kge_model.test_step(kge_model, valid_triples, train_pars, info["entity_dict"])
            test_logs.append(metrics)

        return training_logs, test_logs


class TrainPipeline:
    def __init__(self, config, item_vocab, model, optimizer):
        self.config = config
        self.memory: Deque = deque(maxlen=10000)
        self.policy_net = Net()
        self.target_net = Net()
        self.TARGET_UPDATE = 100
        self.BATCH_SIZE = 10

    def tmp_Q_eps_greedy(self, state, actions):
        epsilon = 0.3
        state = torch.tensor(state, dtype=torch.float)
        out = self.policy_net.forward(state)
        out = out.detach().numpy()
        coin = random.random()
        if coin < epsilon:
            return actions[np.random.choice(range(len(actions)))]
        else:
            return actions[np.argmax(out)]

    def memory_sampling(self, memory: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mini_batch = random.sample(memory, self.BATCH_SIZE)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            t_state, t_action, t_reward, t_next_state, t_done = transition
            s_lst.append(t_state)
            a_lst.append([t_action])
            r_lst.append([t_reward])
            s_prime_lst.append(t_next_state)
            done_mask_lst.append([t_done])
        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def optimize_model(self, memory: Tensor):
        GAMMA = 0.1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_sampling(memory)
        state_action_values = self.policy_net(state_batch)
        next_state_values = self.target_net(next_state_batch)
        max_val_list = []
        for next_state_value in next_state_values:
            max_val = max(next_state_value).tolist()
            max_val_list.append(max_val)
        expected_state_action_values = state_action_values.tolist()
        for i in range(len(state_action_values)):
            action = action_batch[i]
            expected_state_action_values[i][action] = (max_val_list[i] * GAMMA) + reward_batch[i]
        expected_state_action_values = torch.tensor(expected_state_action_values)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # print('loss', loss)
        optimizer = optim.RMSprop(self.policy_net.parameters())
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def run(self):
        simulator = Simulator(config=self.config, mode="train")
        num_users = len(simulator)
        total_step_count = 0
        for e in range(self.config.epochs):
            for u in range(num_users):
                user_id, item_ids, rates = simulator.get_user_data(u)
                candidates = []
                done = False
                print("user_id:", user_id)
                for t, (item_id, rate) in enumerate(zip(item_ids, rates)):
                    if t == len(item_ids) - 1:
                        done = True
                    print("t", t, "item_id", item_id, "rate", rate)
                    # TODO
                    # Embed item using GCN Algorithm1 line 6 ~ 7
                    item_idx = item_id
                    embedded_item_state = self.model.forward_gcn(item_idx)  # (50)
                    embedded_user_state = self.model(item_idx)  # (20)

                    # TODO
                    # Candidate selection and embedding
                    if rate > self.config.threshold:
                        n_hop_dict = self.model.get_n_hop(item_id)
                        candidates.extend(n_hop_dict[1])
                        candidates = list(set(candidates))  # Need to get rid of recommended items

                    candidates_embeddings = self.model.forward_gcn(torch.tensor(candidates))
                    print("candidate shape:", candidates_embeddings.shape)
                    # candidates_embeddings = item_ids  # Embed each item in n_hop_dict using each item's n_hop_dict
                    # candidates_embeddings' shape = (# of candidates, config.item_embed_dim)

                    # Recommendation using epsilon greedy policy
                    recommend_item_id = self.tmp_Q_eps_greedy(state=embedded_user_state, actions=candidates_embeddings)
                    reward = simulator.step(user_id, recommend_item_id)

                    # TODO
                    # Q learning
                    # Store transition to buffer
                    state, action, reward, next_state, done = (
                        embedded_state,
                        recommend_item_id,
                        reward,
                        tmp_state_embed(x.append(recommend_item_id)),
                        done,
                    )
                    Tuple = (state, action, reward, next_state, done)
                    self.memory.append(Tuple)
                    # target update
                    total_step_count += 1
                    if total_step_count % self.TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    if len(self.memory) > 100:
                        self.optimize_model(self.memory)
