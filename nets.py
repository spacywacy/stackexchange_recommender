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

