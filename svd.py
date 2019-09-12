import os
import utils
import sqlite3
import nets
import numpy as np



def create_tables(conn):
	tnames = ['train','test']
	sql_snippets = [
		#classify_train
		'''
		CREATE TABLE {} (
		user_id integer,
		item_id integer,
		label integer
		);
		'''.format(tnames[0]),

		#classify_test
		'''
		CREATE TABLE {} (
		user_id integer,
		item_id integer,
		label integer
		);
		'''.format(tnames[1]),
	]
	for tname, sql_snippet in zip(tnames, sql_snippets):
		utils.create_table(conn, tname, sql_snippet, auto_at_0=True)

class pair_builder():

	def __init__(self, groups, dump_tname, conn, name):
		#input
		self.groups = groups
		self.items = []

		#output
		self.pairs = []

		#db
		self.dump_tname = dump_tname
		self.db_name = '{}.db'.format(name)
		self.conn = conn

		#meta
		self.neg_sample_size = 6
		self.neg_sample_adjust = .75

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
			#user_emb = utils.get_emb(self.conn, self.user_tname, user_id, use_web_id=True, return_str=True)
			for item_id in item_ids:
				self.pos_pairs(user_id, item_id, item_ids)
				self.neg_pairs(user_id, item_ids)
		if self.conn:
			self.dump_pairs()

	def pos_pairs(self, user_id, item_id, context):
		#item_emb = utils.get_emb(self.conn, self.item_tname, item_id, return_str=True)
		self.pairs.append([user_id, item_id, 1])

	def neg_pairs(self, user_id, context):
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


class svd_test():

	def __init__(self):
		#io & db
		self.name = 'svd_test'
		self.db_name = 'svd_test.db'
		self.fav_path = 'storage/cla_01_fav_lookup.csv'
		self.conn = sqlite3.connect('storage/{}'.format(self.db_name))
		create_tables(self.conn)

		#data
		self.train_groups = {}
		self.test_groups = {}

		#meta
		self.test_size = 0.2

	def do_svd(self):
		self.load_groups()
		pair_builder(self.train_groups, 'train', self.conn, self.name).get_pairs()
		pair_builder(self.test_groups, 'test', self.conn, self.name).get_pairs()

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

	



svd_test().do_svd()





































