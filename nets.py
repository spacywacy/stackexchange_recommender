import torch
import torch.nn as nn
import torch.nn.functional as F

class skip_gram(nn.Module):

	def __init__(self, n_items, name=None, emb_dim=10):
		super().__init__()
		self.embeddings = nn.Embedding(n_items, emb_dim)
		self.fc = nn.Linear(emb_dim, n_items)
		self.softmax = nn.Softmax(-1)
		self.name = name

	def forward(self, item_ids):
		z = self.embeddings(item_ids)
		z = self.fc(z)
		z = self.softmax(z)
		return z

class glove(nn.Module):

	def __init__(self, n_items, emb_dim=10):
		super().__init__()
		self.embeddings = nn.Embedding(n_items, emb_dim)
		self.bias = nn.Embedding(n_items, 1)

	def forward(self, item_ids, context_ids):
		item_emb = self.embeddings(item_ids)
		context_emb = self.embeddings(context_ids)
		item_bias = self.bias(item_ids).squeeze(-1)
		context_bias = self.bias(context_ids).squeeze(-1)
		dot = torch.sum(item_emb * context_emb, dim=1)
		z = dot + item_bias + context_bias
		return z

