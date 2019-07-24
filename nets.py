import torch
import torch.nn as nn
import torch.nn.functional as F

class bilinear(nn.Module):

	def __init__(self, n_words, emb_dim=100, name=None):
		super().__init__()
		self.word_embeddings = nn.Embedding(n_words, emb_dim)
		self.fc = nn.Linear(emb_dim, 1)
		self.neg_sample = True
		self.name = name

	def forward(self, word_ids, context_ids):
		word_emb = self.word_embeddings(word_ids)
		context_emb = self.word_embeddings(context_ids)
		z = (word_emb * context_emb)
		z = torch.sigmoid(self.fc(z))
		return z

class bias_bilinear(nn.Module):

	def __init__(self, n_words, emb_dim=100, name=None):
		super().__init__()
		self.word_embeddings = nn.Embedding(n_words, emb_dim)
		self.bias = nn.Embedding(n_words, 1)
		self.fc = nn.Linear(emb_dim, 1, bias=False)
		self.neg_sample = True
		self.name = name
		print('bias')

	def forward(self, word_ids, context_ids):
		word_emb = self.word_embeddings(word_ids)
		context_emb = self.word_embeddings(context_ids)
		z = word_emb * context_emb
		z = self.fc(z)
		bias_w = self.bias(word_ids)
		bias_c = self.bias(context_ids)
		z = z + bias_w + bias_c
		z = torch.sigmoid(z)
		return z


class bias_bilinear_(nn.Module):

	def __init__(self, n_words, emb_dim=100, name=None):
		super().__init__()
		self.word_embeddings = nn.Embedding(n_words, emb_dim)
		self.bias = nn.Embedding(n_words, emb_dim)
		self.fc = nn.Linear(emb_dim, 1, bias=False)
		self.neg_sample = True
		self.name = name

	def forward(self, word_ids, context_ids):
		word_emb = self.word_embeddings(word_ids)
		context_emb = self.word_embeddings(context_ids)
		z = word_emb * context_emb
		bias_w = self.bias(word_ids)
		bias_c = self.bias(context_ids)
		z = z + bias_w + bias_c
		z = self.fc(z)
		z = torch.sigmoid(z)
		return z




class bias_bilinear_naive(nn.Module):

	def __init__(self, n_words, emb_dim=100, name=None):
		super().__init__()
		self.word_embeddings = nn.Embedding(n_words, emb_dim)
		self.fc = nn.Linear(emb_dim, 1)
		self.word_bias = nn.Parameter(torch.ones(1))
		self.con_bias = nn.Parameter(torch.ones(1))
		self.neg_sample = True
		self.name = name

	def forward(self, word_ids, context_ids):
		word_emb = self.word_embeddings(word_ids) + self.word_bias
		context_emb = self.word_embeddings(context_ids) + self.con_bias
		z = (word_emb * context_emb)
		z = torch.sigmoid(self.fc(z))
		return z

class simple_emb(nn.Module):

	def __init__(self, n_words, emb_dim=100, name=None):
		super().__init__()
		self.emb_dim = emb_dim
		self.word_embeddings = nn.Embedding(n_words, emb_dim)
		self.fc = nn.Linear(emb_dim, n_words)
		self.softmax = nn.Softmax(-1)
		self.neg_sample = False
		self.name = name

	def forward(self, word_ids):
		z = self.word_embeddings(word_ids)
		z = self.fc(z)
		z = self.softmax(z)
		return z

