import utils
import os
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from nets import element_product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class classifier():

	def __init__(self, name, data_dir, db_name, neg_sample_size, rep_dim, test_size=0.2):
		#io
		self.data_dir = data_dir
		fav_fname = '{}_fav_lookup.csv'.format(name)
		self.fav_path = os.path.join(self.data_dir, fav_fname)
		model_fname = '{}_classifier.pickle'.format(name)
		self.model_path = os.path.join('bin', model_fname)

		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		self.train_tname = '{}_classify_train'.format(name)
		self.test_tname = '{}_classify_test'.format(name)
		self.user_tname = '{}_users_train'.format(name)
		self.item_tname = '{}_items_train'.format(name)

		#meta
		self.neg_sample_size = neg_sample_size
		self.neg_sample_adjust = .75
		self.test_size = test_size

		#data prep
		self.groups = []
		self.items = []
		self.pairs_train = []
		self.pairs_test = []
		
		#data
		self.user_train = []
		self.item_train = []
		self.label_train = []
		self.user_test = []
		self.item_test = []
		self.label_test = []

		#classifier
		self.criterion = nn.BCELoss()
		self.rep_dim = rep_dim

	def build_dataset(self):
		self.load_groups()
		self.counts_and_probs()
		self.get_pairs()
		self.dump_pairs()
		print('train size:', len(self.pairs_train))
		print('test size:', len(self.pairs_test))

	def load_data(self):
		self.load_train()
		self.load_test()

	def build_classifier(self):
		self.train_classifier()
		self.conn.close()

	def load_groups(self):
		with open(self.fav_path, 'r') as f:
			for line in f:
				row = line[:-1].split(',')
				user_web_id = row[0]
				item_db_ids = [int(x) for x in row[1:]]
				if len(item_db_ids) == 0:
					continue
				self.groups.append([user_web_id, item_db_ids])
				self.items += item_db_ids

	def alter_probs(self):
		#denominator
		denom = 0.0
		for prob in self.item_probs:
			denom += prob**self.neg_sample_adjust

		#adjust probs
		self.item_probs = [(x**self.neg_sample_adjust)/denom for x in self.item_probs]

	def counts_and_probs(self):
		#get total n_items
		self.n_items = len(self.items)

		#get unique items & item counts
		self.items, counts = np.unique(self.items, return_counts=True)
		item_counts = sorted(zip(self.items, counts), key=lambda kv: int(kv[0]), reverse=False)
		self.items = [x[0] for x in item_counts]
		counts = [x[1] for x in item_counts]
		self.n_unique_items = len(self.items)

		#get item probs
		probs = [x/self.n_items for x in counts]
		self.item_probs = dict(zip(self.items, probs))
		self.item_probs = [self.item_probs[x] for x in self.items]
		if self.neg_sample_adjust != 1:
			self.alter_probs()

		print('finished getting probs')
		print('unique items:', self.n_unique_items)

	def get_pairs(self):
		for user_web_id, item_db_ids in self.groups:
			self.train_test_split(user_web_id, item_db_ids)

	def train_test_split(self, user_web_id, item_db_ids):
		user_emb = utils.get_emb(self.conn, self.user_tname, user_web_id, use_web_id=True, return_str=True)
		n_test = round(len(item_db_ids) * self.test_size)
		test_items = np.random.choice(item_db_ids, n_test)
		for item_id in item_db_ids:
			if item_id in test_items:
				self.pos_pairs(user_web_id, user_emb, item_id, item_db_ids, False)
			else:
				self.pos_pairs(user_web_id, user_emb, item_id, item_db_ids, True)

	def pos_pairs(self, user_id, user_emb, item_id, context, train):
		item_emb = utils.get_emb(self.conn, self.item_tname, item_id, return_str=True)
		if train:
			self.pairs_train.append([user_emb, item_emb, 1])
		else:
			self.pairs_test.append([user_emb, item_emb, 1])
		self.neg_pairs(user_id, user_emb, context, train)

	def neg_pairs(self, user_id, user_emb, context, train):
		picked = []
		for _ in range(self.neg_sample_size):
			neg_item = ''
			dont_pick = ['', user_id] + context + picked

			while neg_item in dont_pick:
				neg_item = int(np.random.choice(self.items, 1, p=self.item_probs)[0])

			picked.append(neg_item)
			neg_item_emb = utils.get_emb(self.conn, self.item_tname, neg_item, return_str=True)
			if train:
				self.pairs_train.append([user_emb, neg_item_emb, 0])
			else:
				self.pairs_test.append([user_emb, neg_item_emb, 0])

	def dump_pairs(self):
		cols = ['user_rep', 'item_rep', 'label']

		#insert train set
		cursor = self.conn.cursor()
		for pair in self.pairs_train:
			utils.insert_row(cursor, self.train_tname, pair, cols=cols)
		self.conn.commit()
		cursor.close()

		#insert test set
		cursor = self.conn.cursor()
		for pair in self.pairs_test:
			utils.insert_row(cursor, self.test_tname, pair, cols=cols)
		self.conn.commit()
		cursor.close()

	def load_train(self):
		cursor = self.conn.cursor()
		cols = ['user_rep', 'item_rep', 'label']
		utils.read_table(cursor, self.train_tname, cols=cols)
		for row in cursor:
			user_rep_vec = [float(x) for x in row[0].split(',')]
			item_rep_vec = [float(x) for x in row[1].split(',')]
			self.user_train.append(user_rep_vec)
			self.item_train.append(item_rep_vec)
			self.label_train.append(row[-1])
		cursor.close()
		self.user_train = torch.tensor(self.user_train, dtype=torch.float)
		self.item_train = torch.tensor(self.item_train, dtype=torch.float)
		self.label_train = torch.tensor(self.label_train, dtype=torch.float)
		self.label_train = self.label_train.unsqueeze(1)

	def load_test(self):
		cursor = self.conn.cursor()
		cols = ['user_rep', 'item_rep', 'label']
		utils.read_table(cursor, self.test_tname, cols=cols)
		for row in cursor:
			user_rep_vec = [float(x) for x in row[0].split(',')]
			item_rep_vec = [float(x) for x in row[1].split(',')]
			self.user_test.append(user_rep_vec)
			self.item_test.append(item_rep_vec)
			self.label_test.append(row[-1])
		cursor.close()
		self.user_test = torch.tensor(self.user_test, dtype=torch.float)
		self.item_test = torch.tensor(self.item_test, dtype=torch.float)
		self.label_test = torch.tensor(self.label_test, dtype=torch.float)
		self.label_test = self.label_test.unsqueeze(1)

	def train_classifier(self):
		net = element_product(rep_dim=self.rep_dim)
		learning_rate = 0.001
		epoch = 10000
		optimizer = optim.Adam(net.parameters(), lr=learning_rate)
		for i in range(epoch):
			optimizer.zero_grad()
			output = net(self.user_train, self.item_train)
			loss = self.criterion(output, self.label_train)
			print('epoch: {}, loss: {}'.format(str(i+1), str(float(loss))))
			loss.backward()
			optimizer.step()
		utils.pickle_dump(self.model_path, net)

	def evaluation(self):
		net = utils.pickle_load(self.model_path)
		output = net(self.user_test, self.item_test)
		loss = self.criterion(output, self.label_test)
		print('cross entropy:', float(loss))

		cutoff = 0.25
		predicted = []
		actual = []
		for y_hat, y in zip(output, self.label_test):

			#make prediction
			prob = float(y_hat)
			label = int(y)
			if prob > cutoff:
				label_hat = 1
			else:
				label_hat = 0
			#print(label_hat, label)
			predicted.append(label_hat)
			actual.append(label)

		accuracy = metrics.accuracy_score(actual, predicted)
		recall = metrics.recall_score(actual, predicted)
		precision = metrics.precision_score(actual, predicted)
		print('accuracy:', accuracy)
		print('recall:', recall)
		print('precision:', precision)



























