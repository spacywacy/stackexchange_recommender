import utils
import os
import sqlite3
import numpy as np
from sklearn import metrics
from nets import element_product
from nets import concat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from new_build_dataset import emb_pair_builder
from train_emb import emb_trainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


class pair_builder():

	def __init__(self, groups, dump_tname, conn, name, drop_prev_pairs=True):
		#input
		self.groups = groups
		self.items = []

		#output
		self.pairs = []

		#db
		self.dump_tname = dump_tname
		self.db_name = '{}.db'.format(name)
		self.user_tname = '{}_users_train'.format(name)
		self.item_tname = '{}_items_train'.format(name)
		self.conn = conn

		#meta
		self.neg_sample_size = 6
		self.neg_sample_adjust = .75

		#drop previously pairs
		if drop_prev_pairs:
			utils.delete_from_table(conn, dump_tname)

	def alter_probs(self):
		#denominator
		denom = 0.0
		for prob in self.item_probs:
			denom += prob**self.neg_sample_adjust

		#adjust probs
		self.item_probs = [(x**self.neg_sample_adjust)/denom for x in self.item_probs]

	def counts_and_probs(self):
		#get seperate items
		for user_id, item_ids in self.groups:
			self.items += item_ids

		#get unique items & item counts
		self.n_items = len(self.items)
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
		self.counts_and_probs()
		for user_id, item_ids in self.groups:
			user_emb = utils.get_emb(self.conn, self.user_tname, user_id, use_web_id=True, return_str=True)
			for item_id in item_ids:
				self.pos_pairs(user_id, user_emb, item_id, item_ids)
				self.neg_pairs(user_id, user_emb, item_ids)
		if self.conn:
			self.dump_pairs()

	def pos_pairs(self, user_id, user_emb, item_id, context):
		item_emb = utils.get_emb(self.conn, self.item_tname, item_id, return_str=True)
		self.pairs.append([user_id, item_id, user_emb, item_emb, 1])

	def neg_pairs(self, user_id, user_emb, context):
		picked = []
		for _ in range(self.neg_sample_size):
			neg_item = ''
			dont_pick = ['', user_id] + context + picked

			while neg_item in dont_pick:
				neg_item = int(np.random.choice(self.items, 1, p=self.item_probs)[0])

			picked.append(neg_item)
			neg_item_emb = utils.get_emb(self.conn, self.item_tname, neg_item, return_str=True)
			self.pairs.append([user_id, neg_item, user_emb, neg_item_emb, 0])

	def dump_pairs(self):
		cols = ['user_web_id', 'item_db_id', 'user_rep', 'item_rep', 'label']
		cursor = self.conn.cursor()
		for pair in self.pairs:
			utils.insert_row(cursor, self.dump_tname, pair, cols=cols)
		self.conn.commit()
		cursor.close()
		print('table name: {}, rows: {}'.format(self.dump_tname, len(self.pairs)))


class user_rep():

	def __init__(self, groups, conn, name):
		#input
		self.groups = groups

		#db
		self.conn = conn
		self.items_tname = '{}_items_train'.format(name)
		self.users_tname = '{}_users_train'.format(name)

		#merge user buffer data with training data
		cursor = self.conn.cursor()
		buffer_users_tname = '{}_users_buffer'.format(name)
		utils.move_data(cursor, buffer_users_tname, self.users_tname)
		self.conn.commit()
		cursor.close()

	def build_user_rep(self):
		i = 0
		for user_id, item_ids in self.groups:
			self.single_user(user_id, item_ids, i)
			i+=1

	def single_user(self, user_id, item_ids, i):
		#check if user has favs
		if len(item_ids)==0:
			print('{}th user({}) has no favorites'.format(i, user_id))
			return

		#get embedding
		print('getting rep for {}th user: {}'.format(i, user_id))
		emb_arr = []
		for item_id in item_ids:
			emb = utils.get_emb(self.conn, self.items_tname, item_id)
			emb_arr.append(emb)

		emb_arr = np.array(emb_arr)
		user_rep = np.mean(emb_arr, axis=0)
		user_rep = [','.join([str(x) for x in user_rep])]

		#insert embedding
		cursor = self.conn.cursor()
		utils.update_table(cursor, user_rep, 'web_id', user_id, 'embedding', self.users_tname)
		self.conn.commit()
		cursor.close()


class user_rep_emb():

	def __init__(self, by_user, conn, name, drop_prev_pairs=True):
		#input
		self.name = name
		self.fav_lookup = by_user

		#db
		self.conn = conn
		self.db_name = '{}.db'.format(name)
		self.user_tname = '{}_users_train'.format(name)
		self.userpairs_tname = '{}_userpairs_train'.format(name)

		#io
		self.data_dir = 'storage'
		self.by_fav_fname = '{}_user_by_fav.csv'.format(name)
		self.by_fav_path = os.path.join(self.data_dir, self.by_fav_fname)

		#merge user buffer data with training data
		cursor = self.conn.cursor()
		buffer_user_tname = '{}_users_buffer'.format(name)
		utils.move_data(cursor, buffer_user_tname, self.user_tname)
		self.conn.commit()
		cursor.close()

		#data
		self.by_fav = {} #{item_db_id: [user_web_id]}

		#meta
		self.neg_sample_size = 6

		#drop user pairs from previous run
		if drop_prev_pairs:
			utils.delete_from_table(conn, self.userpairs_tname)

	def build_user_rep(self):
		self.reverse_lookup()
		self.build_pairs()
		self.train_emb()


	def reverse_lookup(self):
		for user_web_id, item_db_ids in self.fav_lookup:
			user_db_id = utils.user_id_web2db(self.conn, self.user_tname, user_web_id)
			for item_db_id in item_db_ids:
				self.by_fav[item_db_id] = self.by_fav.get(item_db_id, []) + [user_db_id]
		self.by_fav = self.by_fav.values()

		#write by fav to file
		with open(self.by_fav_path, 'w') as f:
			for group in self.by_fav:
				line = ','.join([str(x) for x in group]) + '\n'
				f.write(line)

	def build_pairs(self):
		print(self.db_name)
		print(self.userpairs_tname)
		emb_pair = emb_pair_builder(self.name,
									self.data_dir,
									self.db_name,
									self.userpairs_tname,
									self.by_fav_fname,
									self.neg_sample_size)
		emb_pair.create_pairs()

	def train_emb(self):
		lr = 0.01
		batch_size = 7000
		n_epoch = 1500
		emb_dim = 4
		emb_train = emb_trainer(
				lr,
				self.db_name,
				batch_size,
				n_epoch,
				emb_dim,
				self.userpairs_tname,
				self.user_tname,
				'id',
				name = self.name
			)
		emb_train.train_loop()



class classifier():

	def __init__(self, name, data_dir, db_name, neg_sample_size, rep_dim, test_size=0.2):
		#io
		self.data_dir = data_dir
		fav_fname = '{}_fav_lookup.csv'.format(name)
		self.fav_path = os.path.join(self.data_dir, fav_fname)
		model_fname = '{}_classifier.pickle'.format(name)
		self.model_path = os.path.join('bin', model_fname)
		fav_train_fname = '{}_fav_train.csv'.format(name)
		self.fav_train_path = os.path.join(self.data_dir, fav_train_fname)
		self.scaler_path = os.path.join(self.data_dir, '{}_scalers.pickle'.format(name))

		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		self.train_tname = '{}_classify_train'.format(name)
		self.test_tname = '{}_classify_test'.format(name)
		self.user_tname = '{}_users_train'.format(name)
		self.item_tname = '{}_items_train'.format(name)

		#meta
		self.name = name
		self.test_size = test_size
		self.norm_input = False

		#data prep
		self.train_groups = {}
		self.test_groups = {}

		#classifier
		self.criterion = nn.BCELoss()
		self.rep_dim = rep_dim

	def build_dataset(self):
		self.load_groups()
		self.write_train_favs()
		#user_rep(self.train_groups, self.conn, self.name).build_user_rep()
		user_rep_emb(self.train_groups, self.conn, self.name).build_user_rep()
		pair_builder(self.train_groups, self.train_tname, self.conn, self.name).get_pairs() #build training set
		pair_builder(self.test_groups, self.test_tname, self.conn, self.name).get_pairs() #build testing set

	def build_classifier(self):
		self.user_train, self.item_train, self.label_train = self.load_from_db(self.train_tname)
		self.train_classifier()

	def load_groups(self): #does train test split while loading data
		with open(self.fav_path, 'r') as f:
			for line in f:
				row = line[:-1].split(',')
				user_web_id = row[0]
				item_db_ids = [int(x) for x in row[1:]]
				if len(item_db_ids) == 0:
					continue
				self.train_test_split(user_web_id, item_db_ids)

		self.train_groups = self.train_groups.items()
		self.test_groups = self.test_groups.items()

	def train_test_split(self, user_id, item_ids):
		n_test = round(len(item_ids) * self.test_size)
		test_items = np.random.choice(item_ids, n_test)
		for item_id in item_ids:
			if item_id in test_items:
				self.test_groups[user_id] = self.test_groups.get(user_id, []) + [item_id]
			else:
				self.train_groups[user_id] = self.train_groups.get(user_id, []) + [item_id]

	def write_train_favs(self):
		with open(self.fav_train_path, 'w') as f:
			for user_web_id, item_db_ids in self.train_groups:
				line = [user_web_id] + item_db_ids
				line = ','.join([str(x) for x in line]) + '\n'
				f.write(line)

	def load_from_db(self, tname):
		cursor = self.conn.cursor()
		cols = ['user_rep', 'item_rep', 'label']
		user_xs, item_xs, labels = [], [], []
		utils.read_table(cursor, tname, cols=cols)
		for row in cursor:
			user_rep_vec = [float(x) for x in row[0].split(',')]
			item_rep_vec = [float(x) for x in row[1].split(',')]

			#experiment with normalization
			#user_rep_vec = user_rep_vec/np.linalg.norm(user_rep_vec)
			#item_rep_vec = item_rep_vec/np.linalg.norm(item_rep_vec)


			user_xs.append(user_rep_vec)
			item_xs.append(item_rep_vec)
			labels.append(row[-1])
		cursor.close()

		if self.norm_input:
			#exp w/ norm
			#user_xs = normalize(np.array(user_xs), axis=1)
			#item_xs = normalize(np.array(item_xs), axis=1)
			user_scaler = StandardScaler()
			item_scaler = StandardScaler()
			user_scaler.fit(user_xs)
			item_scaler.fit(item_xs)
			user_xs = user_scaler.transform(user_xs)
			item_xs = item_scaler.transform(item_xs)
			utils.pickle_dump(self.scaler_path, (user_scaler, item_scaler))

		user_xs = torch.tensor(user_xs, dtype=torch.float)
		item_xs = torch.tensor(item_xs, dtype=torch.float)
		labels = torch.tensor(labels, dtype=torch.float)
		labels = labels.unsqueeze(1)
		return user_xs, item_xs, labels

	def train_classifier(self):
		#net = element_product(rep_dim=self.rep_dim)
		net = concat(4, 30)
		learning_rate = 0.001
		#learning_rate = 0.005
		epoch = 4000
		optimizer = optim.Adam(net.parameters(), lr=learning_rate)
		for i in range(epoch):
			optimizer.zero_grad()
			#print(self.user_train)
			#print(self.item_train)
			output = net(self.user_train, self.item_train)
			loss = self.criterion(output, self.label_train)
			print('epoch: {}, loss: {}'.format(str(i+1), str(float(loss))))
			loss.backward()
			optimizer.step()
		utils.pickle_dump(self.model_path, net)

	def evaluation(self):
		self.user_test, self.item_test, self.label_test = self.load_from_db(self.test_tname)
		net = utils.pickle_load(self.model_path)
		output = net(self.user_test, self.item_test)
		loss = self.criterion(output, self.label_test)
		print('cross entropy:', float(loss))

		cutoff = 0.5
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

	def prob_rank_all(self):
		k = 10
		self.user_test, self.item_test, self.label_test = self.load_from_db(self.test_tname)
		net = utils.pickle_load(self.model_path)
		output = net(self.user_test, self.item_test)
		top_items = sorted(zip(output, self.label_test), key=lambda x: x[0], reverse=True)[:k]
		#for prob, label in top_items:
			#print(round(float(prob), 3), float(label))

		probs = [float(x[0]) for x in top_items]
		labels = [float(x[1]) for x in top_items]
		ap = metrics.average_precision_score(labels, probs)
		print('average precision @{}: {}'.format(k, ap))

	def prob_rank(self, user_web_id, k):
		#get data from db
		user_xs, item_xs, labels = [], [], []
		cursor = self.conn.cursor()
		sql_ = '''
				SELECT user_rep, item_rep, label
				FROM {}
				WHERE id != -1
				AND user_web_id = ?
				ORDER BY id;
			   '''.format(self.test_tname)
		cursor.execute(sql_, [user_web_id])
		for row in cursor:
			user_rep_vec = [float(x) for x in row[0].split(',')]
			item_rep_vec = [float(x) for x in row[1].split(',')]
			user_xs.append(user_rep_vec)
			item_xs.append(item_rep_vec)
			labels.append(row[-1])
		cursor.close()
		if len(user_xs)==0:
			return None

		#exp w/ norm
		if self.norm_input:
			user_scaler, item_scaler = utils.pickle_load(self.scaler_path)
			user_xs = user_scaler.transform(user_xs)
			item_xs = item_scaler.transform(item_xs)
			print(np.array(user_xs))

		user_xs = torch.tensor(user_xs, dtype=torch.float)
		item_xs = torch.tensor(item_xs, dtype=torch.float)
		labels = torch.tensor(labels, dtype=torch.float)
		labels = labels.unsqueeze(1)

		#get average precision
		net = utils.pickle_load(self.model_path)
		net.eval()
		output = net(user_xs, item_xs)
		top_items = sorted(zip(output, labels), key=lambda x: x[0], reverse=True)[:k]
		probs = [float(x[0]) for x in top_items]
		labels = [float(x[1]) for x in top_items]
		ap = metrics.average_precision_score(labels, probs)
		if not np.isnan(ap):
			#print('average precision @{} for user {}: {}'.format(k, user_web_id, ap))
			return ap
		else:
			return None

	def prob_rank_by_user(self):
		#get users
		cursor = self.conn.cursor()
		sql_ = 'SELECT web_id FROM {} WHERE id != -1;'.format(self.user_tname)
		cursor.execute(sql_)
		user_web_ids = [x[0] for x in cursor.fetchall()]
		cursor.close()

		#get MAP@K
		k = 5
		APs = []
		for user in user_web_ids:
			ap = self.prob_rank(user, k)
			if ap:
				APs.append(ap)
		map_at_k = np.array(APs).mean()
		print('MAP @ K:', map_at_k)

		#plt.hist(APs, bins=20)
		#plt.show()

	def recommend(self, user_web_id, k):
		#read item emb
		item_info = []
		item_xs = []
		cursor = self.conn.cursor()
		sql_ = '''
				SELECT embedding, title, link
				FROM {}
				WHERE id != -1
				ORDER BY score desc
				LIMIT 2000;
			   '''.format(self.item_tname)
		cursor.execute(sql_)
		for row in cursor:
			item_info.append(row[1:])
			item_emb = [float(x) for x in row[0].split(',')]
			item_xs.append(item_emb)
		cursor.close()
		item_xs = torch.tensor(item_xs, dtype=torch.float)

		#get user emb
		cursor = self.conn.cursor()
		sql_ = 'SELECT embedding FROM {} WHERE web_id=?;'.format(self.user_tname)
		cursor.execute(sql_, [user_web_id])
		user_emb = cursor.fetchall()[0][0]
		user_emb = [float(x) for x in user_emb.split(',')]
		cursor.close()
		user_xs = [user_emb for x in range(len(item_xs))]
		user_xs = torch.tensor(user_xs, dtype=torch.float)

		#model pass
		net = utils.pickle_load(self.model_path)
		net.eval()
		output = net(user_xs, item_xs)
		top_items = sorted(zip(output, item_info), key=lambda x: x[0], reverse=True)[:k]
		for item in top_items:
			print(round(float(item[0]),2), item[1][0], item[1][1])
































