import numpy as np
import pickle
import torch
import torch.optim as optim
from model import GCN_GRU, Net
from env import Simulator, Config
from dataloader import *

def pretrain_embedding(config, entity_vocab, relation_vocab, model, optimizer):
	model.train()

	dataloader = get_TransE_dataloader(config, entity_vocab, relation_vocab)
	for epoch in range(200):
		total_loss = 0
		for positive_triples, negative_triples in dataloader:	
			optimizer.zero_grad()
			loss = model.TransE_forward(positive_triples, negative_triples)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
		print('TransE epoch', epoch, 'loss', total_loss)


def train(config, item_vocab, model, optimizer):
	memory = deque(maxlen=10000)
	policy_net = Net()
	target_net = Net()
	TARGET_UPDATE = 100
	BATCH_SIZE = 10

	def tmp_Q_eps_greedy(state, actions):
		epsilon = 0.3
		state = torch.tensor(state, dtype=torch.float)
		out = policy_net.forward(state)
		out = out.detach().numpy()
		coin = random.random()
		if coin < epsilon:
			return actions[np.random.choice(range(len(actions)))]
		else:
			return actions[np.argmax(out)]
	
	def memory_sampling(memory):
		mini_batch = random.sample(memory, BATCH_SIZE)
		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

		for transition in mini_batch:
			t_state, t_action, t_reward, t_next_state, t_done = transition
			s_lst.append(t_state)
			a_lst.append([t_action])
			r_lst.append([t_reward])
			s_prime_lst.append(t_next_state)
			done_mask_lst.append([t_done])
		return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)

	def optimize_model(memory):
		state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory_sampling(memory)
		state_action_values = policy_net(state_batch)    
		next_state_values = target_net(next_state_batch)
		for next_state_value in next_state_values:
			max_val = max(next_state_value).tolist()
			max_val_list.append(max_val)
		expected_state_action_values = state_action_values.tolist()
		for i in range(len(state_action_values)):
			action = action_batch[i]
			expected_state_action_values[i][action] = (max_val_list[i] * GAMMA) + reward_batch[i]
		expected_state_action_values = torch.tensor(expected_state_action_values)
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
		#print('loss', loss)
		optimizer = optim.RMSprop(self.policy_net.parameters())
		optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()

	simulator = Simulator(config=config, mode='train')
	num_users = len(simulator)
	total_step_count = 0
	for e in range(config.epochs):
		for u in range(num_users):
			user_id, item_ids, rates = simulator.get_data(u)
			candidates = []
			done = False
			print('user_id:', user_id)
			for t, (item_id, rate) in enumerate(zip(item_ids, rates)):
				if t == len(item_ids)-1: done = True
				print('t',t,'item_id',item_id,'rate',rate)
				# TODO
				# Embed item using GCN Algorithm1 line 6 ~ 7
				item_idx = item_id
				embedded_item_state = model.forward_GCN(item_idx)	# (50)
				embedded_user_state = model(item_idx)				# (20)

				# TODO
				# Candidate selection and embedding
				if rate > config.threshold:
					n_hop_dict = model.get_n_hop(item_id)
					candidates.extend(n_hop_dict[1])
					candidates = list(set(candidates))      # Need to get rid of recommended items

				candidates_embeddings = model.forward_GCN(torch.tensor(candidates))
				print('candidate shape:',candidates_embeddings.shape)
				# candidates_embeddings = item_ids  # Embed each item in n_hop_dict using each item's n_hop_dict
				# candidates_embeddings' shape = (# of candidates, config.item_embed_dim)

				# Recommendation using epsilon greedy policy
				recommend_item_id = tmp_Q_eps_greedy(state=embedded_user_state, actions=candidates_embeddings)
				reward = simulator.step(user_id, recommend_item_id)

				# TODO
				# Q learning
				# Store transition to buffer
				state, action, reward, next_state, done = embedded_state, recommend_item_id, reward, tmp_state_embed(x.append(recommend_item_id)), done # done을 어떻게 하지?
				Tuple = (state, action, reward, next_state, done)        
				memory.append(Tuple)
				# target update
				total_step_count+=1
				if total_step_count % TARGET_UPDATE ==0:
					target_net.load_state_dict(policy_net.state_dict())
				if len(memory) > 100:    
					optimize_model(memory)

if __name__ == '__main__':

	with open('./data/movie/entity_vocab.pkl','rb') as f:
		entity_vocab = pickle.load(f)
	with open('./data/movie/item_vocab.pkl','rb') as f:
		item_vocab = pickle.load(f)
	with open('./data/movie/relation_vocab.pkl','rb') as f:
		relation_vocab = pickle.load(f)

	print('| Building Net')
	model = GCN_GRU(Config(), 50, entity_vocab, relation_vocab)
	optimizer = optim.SGD(model.parameters(), lr=0.01)
	
	print('Embedding pretrain by TransE...')
	pretrain_embedding(Config(), entity_vocab, relation_vocab, model, optimizer)

	print('Save embedding_pretrained model...')
	path = './embedding_pretrained.pth'
	torch.save(model.state_dict(),path)
	
	print('Load embedding_pretrained model...')
	path = './embedding_pretrained.pth'
	model.load_state_dict(torch.load(path))

	print('Train...')
	train(Config(), item_vocab, model, optimizer)
