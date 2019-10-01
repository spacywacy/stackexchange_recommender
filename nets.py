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

class glove_item(nn.Module):

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

class glove(nn.Module):

	def __init__(self, n_users, n_items, user_emb_dim, item_emb_dim):
		super().__init__()
		self.user_emb = nn.Embedding(n_users, user_emb_dim)
		self.item_emb = nn.Embedding(n_items, item_emb_dim)
		self.user_bias = nn.Embedding(n_users, 1)
		self.item_bias = nn.Embedding(n_items, 1)
		self.drop_user = nn.Dropout()
		self.drop_item = nn.Dropout()
		#self.sigmoid = nn.Sigmoid()

	def forward(self, item_ids, context_ids):
		user_z = self.drop_user(self.user_emb(item_ids))
		item_z = self.drop_item(self.item_emb(context_ids))
		user_b = self.user_bias(item_ids).squeeze(-1)
		item_b = self.item_bias(context_ids).squeeze(-1)
		dot = torch.sum(user_z * item_z, dim=1)
		z = dot + user_b + item_b
		return z
		#return self.sigmoid(z)

class element_product_(nn.Module):

	def __init__(self, rep_dim=10):
		super().__init__()
		self.fc = nn.Linear(rep_dim, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, user_vec, item_vec):
		z = user_vec * item_vec
		z = self.sigmoid(self.fc(z))
		return z

class element_product(nn.Module):

	def __init__(self, rep_dim=25):
		super().__init__()
		self.fc1 = nn.Linear(rep_dim, rep_dim)
		self.fc2 = nn.Linear(rep_dim, rep_dim)
		self.fc3 = nn.Linear(rep_dim, 1)
		self.dropout = nn.Dropout(p=.5)
		self.sigmoid = nn.Sigmoid()

	def forward(self, user_vec, item_vec):
		z = user_vec * item_vec
		z = self.dropout(self.sigmoid(self.fc1(z)))
		z = self.dropout(self.sigmoid(self.fc2(z)))
		z = self.sigmoid(self.fc3(z))
		return z

class concat(nn.Module):

	def __init__(self, user_dim, item_dim):
		super().__init__()
		dim = user_dim + item_dim
		self.fc1 = nn.Linear(dim, 100)
		self.fc2 = nn.Linear(100, 35)
		self.fc3 = nn.Linear(35, 35)
		self.fc4 = nn.Linear(35, 1)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=.5)

	def forward(self, user_vec, item_vec):
		z = torch.cat([user_vec, item_vec], dim=1)
		z = self.dropout(self.sigmoid(self.fc1(z)))
		z = self.sigmoid(self.fc2(z))
		z = self.sigmoid(self.fc3(z))
		z = self.sigmoid(self.fc4(z))
		return z

class concat_(nn.Module):

	def __init__(self, user_dim, item_dim):
		super().__init__()
		dim = user_dim + item_dim
		self.fc1 = nn.Linear(dim, 100)
		self.fc2 = nn.Linear(100, 35)
		self.fc3 = nn.Linear(35, 35)
		self.fc4 = nn.Linear(35, 1)
		self.dropout = nn.Dropout(p=.5)
		self.sigmoid = nn.Sigmoid()

	def forward(self, user_vec, item_vec):
		z = torch.cat([user_vec, item_vec], dim=1)
		z = self.dropout(self.sigmoid(self.fc1(z)))
		z = self.dropout(self.sigmoid(self.fc2(z)))
		z = self.dropout(self.sigmoid(self.fc3(z)))
		z = self.sigmoid(self.fc4(z))
		return z


























