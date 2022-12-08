# Modified from https://github.com/AlexYangLi/KGCN_Keras

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from config import Config
from utils import pickle_dump, pickle_load


random.seed(14)
np.random.seed(14)


def read_item2entity_file(item2entity_path, item_vocab, entity_vocab):
	'''
		item_vocab : Change indicator in rating file to item index in code
		entity_vocab: Change indicator in knowledge graph file to index in code
	'''
	print(f'Logging Info - Reading item2entity file: {item2entity_path}')
	assert len(item_vocab) == 0 and len(entity_vocab) == 0
	with open(item2entity_path, encoding='utf8') as reader:
		for line in reader:
			item, entity = line.strip().split('\t')
			item = int(item)
			entity = int(entity)
			item_vocab[item] = len(item_vocab)
			entity_vocab[entity] = len(entity_vocab)


def read_rating_file(rating_path, separator, minimum_interactions, user_vocab, item_vocab, entity_vocab):
	'''
		user_vocab: Change user indicator in rating to user index in code
	'''
	print(f'Logging Info - Reading rating file: {rating_path}')
	assert len(user_vocab) == 0 and len(item_vocab) > 0
	user_rating = defaultdict(list)

	# Save rating datas into user_rating dict
	with open(rating_path, encoding='utf8') as reader:
		for idx, line in enumerate(reader):
			if idx == 0:                                            # Ignore first line
				continue
			user, item, rating, timestamp = line.strip().split(separator)[:4]
			user, item, rating = int(user), int(item), float(rating)
			if item in item_vocab:                                  # Ignore item not in KG
				user_rating[user].append((item_vocab[item], rating, timestamp))

	# Remove user who has less interaction than 200 (minimum_interactions)
	# # of remained users has to be 16,525
	users2remove = []
	for k, v in user_rating.items():
		if len(v) < minimum_interactions:
			users2remove.append(k)
	for user2remove in users2remove:
		del user_rating[user2remove]

	# Split train : val : test = 8 : 1 : 1
	remained_users = list(user_rating.keys())
	total_users = len(remained_users)
	random.shuffle(remained_users)
	train_users = remained_users[:int(total_users*0.8)]
	val_users = remained_users[int(total_users*0.8):int(total_users*0.9)]

	# # of interactions has to be 6,711,013
	print('Logging Info - Converting rating file...')
	train_rating_dict, valid_rating_dict, test_rating_dict = {}, {}, {}
	unwatched_item_id_set = set(item_vocab.values())
	for user, rated_items_by_user in user_rating.items():
		if user in train_users:
			dict = train_rating_dict
		elif user in val_users:
			dict = valid_rating_dict
		else:
			dict = test_rating_dict

		user_vocab[user] = len(user_vocab)
		user_id = user_vocab[user]

		dict[user_id] = []
		for item_id, rate, timestamp in rated_items_by_user:
			if item_id in unwatched_item_id_set:
				unwatched_item_id_set.remove(item_id)
			dict[user_id].append([item_id, rate, timestamp])

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

	print(f'Logging Info - num of users: {len(user_vocab)}, num of items: {len(item_vocab)}')
	return train_rating_dict, test_rating_dict, valid_rating_dict


def read_kg(kg_path, entity_vocab, relation_vocab, user_vocab, item_vocab):
	'''
		entity_vocab: Change c indicator in item2entity file to entity index in code
		adj_mat: adjacency matrix of kg
		feat_mat: TransE pretrained feature of kg
	'''
	print(f'Logging Info - Reading kg file: {kg_path}')

	kg = defaultdict(list)
	print('# user:', len(user_vocab), '# item:', len(item_vocab), '# entity:', len(entity_vocab))

	entity_freq = {}
	with open(kg_path, encoding='utf8') as reader:
		for line in reader:
			head, relation, tail = line.strip().split('\t')
			head, tail = int(head), int(tail)
			if tail not in entity_freq:
				entity_freq[tail] = 0
			else:
				entity_freq[tail] += 1
	entity_sorted_key = []
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

	#feat_mat = np.zeros((len(user_vocab)+len(item_vocab), 50))
	with open(kg_path, encoding='utf8') as reader:
		for line in reader:
			head, relation, tail = line.strip().split('\t')
			head, tail = int(head), int(tail)
			if head not in entity_vocab or tail not in entity_vocab:
				continue
			else:
				if relation not in relation_vocab:
					relation_vocab[relation] = len(relation_vocab)

			# Undirected graph
			kg[entity_vocab[head]].append(entity_vocab[tail])
			kg[entity_vocab[tail]].append(entity_vocab[head])

	print('# user:', len(user_vocab), '# item:', len(item_vocab), '# entity:', len(entity_vocab))
	max_entity = max(entity_vocab.values())
	adj_mat = np.zeros((max_entity, max_entity))

	with open(kg_path, encoding='utf8') as reader:
		for line in reader:
			head, relation, tail = line.strip().split('\t')
			head, tail = int(head), int(tail)
			if head in entity_vocab and tail in entity_vocab:
				adj_mat[entity_vocab[head]-1][entity_vocab[tail]-1] = 1
				adj_mat[entity_vocab[tail]-1][entity_vocab[head]-1] = 1
	
	n_hop_kg = {}
	for entity in entity_vocab.values():
		n_hop_kg[entity] = {1: [], 2: []}
		n_hop_kg[entity][1] = kg[entity]
		for t in kg[entity]:
			n_hop_kg[entity][2].extend(kg[t])
		n_hop_kg[entity][2] = list(set(n_hop_kg[entity][2]) - set(n_hop_kg[entity][1]))

	print(f'Logging Info - num of entities: {len(entity_vocab)}, num of relations: {len(relation_vocab)}', '# adj:', adj_mat.sum()/2)
	return n_hop_kg, adj_mat


def process_data(config):
	os.makedirs(config.preprocess_results_dir, exist_ok=True)

	# Sort rating file based on user id and timestamp
	df = pd.read_csv(f'{config.raw_data_dir}/{config.dataset_name}/ratings.csv', delimiter=',')
	df = df.sort_values(by=['userId', 'timestamp'], ascending=[True, True])
	sorted_rating_path = f'{config.raw_data_dir}/{config.dataset_name}/sorted.csv'
	df.to_csv(sorted_rating_path, index=False)

	user_vocab = {}
	item_vocab = {}
	entity_vocab = {}
	relation_vocab = {}

	read_item2entity_file(config.item2entity_path, item_vocab, entity_vocab)
	train_data_dict, val_data_dict, test_data_dict = read_rating_file(sorted_rating_path, config.separator,
																	  config.minimum_interactions,
																	  user_vocab, item_vocab, entity_vocab)
	pickle_dump(f'{config.preprocess_results_dir}/user_vocab.pkl', user_vocab)
	pickle_dump(f'{config.preprocess_results_dir}/item_vocab.pkl', item_vocab)
	pickle_dump(f'{config.preprocess_results_dir}/train_data_dict.pkl', train_data_dict)
	pickle_dump(f'{config.preprocess_results_dir}/val_data_dict.pkl', val_data_dict)
	pickle_dump(f'{config.preprocess_results_dir}/test_data_dict.pkl', test_data_dict)

	n_hop_kg, adj_mat = read_kg(config.kg_path, entity_vocab, relation_vocab, user_vocab, item_vocab)
	pickle_dump(f'{config.preprocess_results_dir}/entity_vocab.pkl', entity_vocab)
	pickle_dump(f'{config.preprocess_results_dir}/relation_vocab.pkl', relation_vocab)
	pickle_dump(f'{config.preprocess_results_dir}/n_hop_kg.pkl', n_hop_kg)
	np.save(f'{config.preprocess_results_dir}/kg_adj_mat.npy', adj_mat)


if __name__ == '__main__':
	process_data(Config())
