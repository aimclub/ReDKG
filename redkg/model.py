import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from utils import pickle_load

class GraphConvolution(nn.Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Linear(in_features, out_features)
		'''
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		'''
		self.adj = torch.FloatTensor(np.load('./data/movie/kg_adj_mat.npy'))

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		#support = torch.mm(input, self.weight)
		support = self.weight(input)
		output = torch.spmm(self.adj, support)
		return output
		'''
		if self.bias is not None:
			return output + self.bias
		else:
			return output
		'''

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'


class GCN_GRU(nn.Module):
	def __init__(self, config, nfeat, entity_vocab, relation_vocab):
		super(GCN_GRU, self).__init__()

		entity_max = max(entity_vocab.values())
		relation_max = max(relation_vocab.values())
		self.n_hop_kg = pickle_load(f'{config.preprocess_results_dir}/n_hop_kg.pkl')

		self.entity_emb = torch.nn.Embedding(entity_max, nfeat)
		uniform_range = 6 / np.sqrt(nfeat)
		self.entity_emb.weight.data.uniform_(-uniform_range, uniform_range)

		self.relation_emb = torch.nn.Embedding(relation_max, nfeat)
		uniform_range = 6 / np.sqrt(nfeat)
		self.relation_emb.weight.data.uniform_(-uniform_range, uniform_range)

		self.gc1 = GraphConvolution(nfeat, nfeat)
		self.gc2 = GraphConvolution(nfeat, nfeat)

		self.h = torch.randn(2,1,20)
		self.gru = nn.GRU(50, 20, 2)
		
		self.criterion = nn.MarginRankingLoss(margin=1.0)

	def get_n_hop(self, entity_id):
		return self.n_hop_kg[entity_id]

	def distance(self, triplets):
		assert triplets.size()[1] == 3
		heads = triplets[:, 0]
		relations = triplets[:, 1]
		tails = triplets[:, 2]
		return (self.entity_emb.weight[heads] + self.relation_emb.weight[relations] - self.entity_emb.weight[tails]).norm(p=1, dim=1)

	def TransE_forward(self, pos_triplet, neg_triplet):
		# -1 to avoid nan for OOV vector
		self.entity_emb.weight.data[:-1, :].div_(self.entity_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

		pos_distance = self.distance(pos_triplet)
		neg_distance = self.distance(neg_triplet)

		target = torch.tensor([-1], dtype=torch.long)

		return self.criterion(pos_distance, neg_distance, target)

	def forward_GCN(self, x):
		# GCN
		out = F.relu(self.gc1(self.entity_emb.weight))
		out = self.gc2(out)
		out = F.log_softmax(out, dim=1)[x]
		return out

	def forward(self, x):
		# GCN
		out = F.relu(self.gc1(self.entity_emb.weight))
		out = self.gc2(out)
		out = F.log_softmax(out, dim=1)[x].reshape(1,1,-1)
	
		# GRU
		out, self.h = self.gru(out, self.h)
		out = out.reshape(-1)
		return out


class GRU(nn.Module):
	def __init__(self, config, nfeat, entity_vocab, relation_vocab):
		super(GRU, self).__init__()

		entity_max = max(entity_vocab.values())
		relation_max = max(relation_vocab.values())
		self.n_hop_kg = pickle_load(f'{config.preprocess_results_dir}/n_hop_kg.pkl')

		self.entity_emb = torch.nn.Embedding(entity_max, nfeat)
		uniform_range = 6 / np.sqrt(nfeat)
		self.entity_emb.weight.data.uniform_(-uniform_range, uniform_range)

		self.relation_emb = torch.nn.Embedding(relation_max, nfeat)
		uniform_range = 6 / np.sqrt(nfeat)
		self.relation_emb.weight.data.uniform_(-uniform_range, uniform_range)

		self.h = torch.randn(2,1,20)
		self.gru = nn.GRU(50, 20, 2)
		
		self.criterion = nn.MarginRankingLoss(margin=1.0)

	def get_n_hop(self, entity_id):
		return self.n_hop_kg[entity_id]

	def distance(self, triplets):
		assert triplets.size()[1] == 3
		heads = triplets[:, 0]
		relations = triplets[:, 1]
		tails = triplets[:, 2]
		return (self.entity_emb.weight[heads] + self.relation_emb.weight[relations] - self.entity_emb.weight[tails]).norm(p=1, dim=1)

	def TransE_forward(self, pos_triplet, neg_triplet):
		# -1 to avoid nan for OOV vector
		self.entity_emb.weight.data[:-1, :].div_(self.entity_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

		pos_distance = self.distance(pos_triplet)
		neg_distance = self.distance(neg_triplet)

		target = torch.tensor([-1], dtype=torch.long)

		return self.criterion(pos_distance, neg_distance, target)

	def forward_GCN(self, x):
		# GCN
		out = F.relu(self.gc1(self.entity_emb.weight))
		out = self.gc2(out)
		out = F.log_softmax(out, dim=1)[x]
		return out

	def forward(self, x):
		x = self.entity_emb.weight[x]
		x = x.reshape(1,1,-1)
	
		# GRU
		out, self.h = self.gru(x, self.h)
		out = out.reshape(-1)
		return out	
	
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(20, 128) 
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
