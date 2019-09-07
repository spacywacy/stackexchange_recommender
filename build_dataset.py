import utils
import requests
import json
import pickle
import os
import sqlite3
import numpy as np
from time import sleep


class stack_api_wrapper():

	def __init__(self, name, db_name, data_dir, n_user_pages, user_pagesize, api_tag):
		#io
		self.key_loc = 'resources/key'
		self.result_dir = data_dir
		self.user_ids_fname = '{}_user_ids.pickle'.format(name)
		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)

		#get key
		with open(self.key_loc, 'r') as f:
			self.key = f.read()

		#utils
		self.site = 'crossvalidated'
		self.n_user_pages = n_user_pages
		self.user_pagesize = user_pagesize
		self.n_items_per_user = 100
		self.delay = 5
		self.api_tag = api_tag

		#db
		self.conn = sqlite3.connect(os.path.join(self.result_dir, db_name))
		self.user_tname = '{}_users_buffer'.format(name)
		self.item_tname = '{}_items_buffer'.format(name)

		#counters
		self.n_users_inserted = 0
		self.n_items_inserted = 0

	def api_call(self):
		self.get_users()
		self.get_questions()
		self.conn.close()


	def get_users(self):
		#request header
		url = 'https://api.stackexchange.com/2.2/users'
		params = {
			'order':'desc',
			'sort':'reputation',
			'site':self.site,
			'pagesize':self.user_pagesize,
			'page':'',
			'key':self.key
		}

		#request loop
		user_ids = []
		for i in range(1, self.n_user_pages+1):
			print('page:', i)
			params['page'] = i

			#get data
			json_items = utils.call_api(url, params)
			user_ids_in_loop = [utils.zero_padding(x['user_id']) for x in json_items]

			#append data
			user_ids += user_ids_in_loop
			self.n_users_inserted += utils.store_users(json_items, self.user_tname, self.conn)

			#delay
			sleep(self.delay)

		#cache user id list
		ids_path = os.path.join(self.result_dir, self.user_ids_fname)
		utils.pickle_dump(ids_path, user_ids)

	def get_questions(self):
		#load user ids
		pickle_path = os.path.join(self.result_dir, self.user_ids_fname)
		user_ids = utils.pickle_load(pickle_path)

		#request header
		params = {
			'order':'desc',
			'sort':'votes',
			'site':self.site,
			'key':self.key
		}
		
		#request loop
		i_user = 1
		for user_id in user_ids:
			print('get favs for {}th user'.format(i_user))
			url = 'https://api.stackexchange.com/2.2/users/{}/{}'.format(user_id, self.api_tag)

			#get & store data
			json_items = utils.call_api(url, params)
			self.n_items_inserted += utils.store_questions(json_items, str(user_id), self.item_tname, self.conn)
			i_user += 1

			#delay
			sleep(self.delay)


class pair_builder():

	def __init__(self, name, data_dir, db_name, neg_sample_size):
		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		self.items_tname = '{}_items_buffer'.format(name)
		self.pairs_tname = '{}_pairs_buffer'.format(name)

		#cache
		self.pairs = [] #item_id, item_id, label
		self.by_user = {} #{user_id: [item_id]}
		self.items = []
		self.n_unique_items = 0
		self.n_items = 0

		#others
		self.neg_sample_size = neg_sample_size
		self.neg_sample_adjust = .75
		self.drop_singles = False
		

	def create_pairs(self):
		self.group_by_user()
		self.counts_and_probs()
		self.get_pairs()
		self.dump_pairs()
		self.conn.close()
		print('Total of {} items, {} pairs'.format(len(self.items), len(self.pairs)))
		print('Inserted {} pairs'.format(self.n_pairs_inserted))

	def load_data(self):
		cursor = self.conn.cursor()
		utils.read_table(cursor, self.items_tname)
		for row in cursor:
			yield row
		cursor.close()

	def group_by_user(self, show=False, debug=False):
		if debug:
			self.items_tname = '_'.join(self.items_tname.split('_')[:-1] + ['train'])

		for row in self.load_data():
			item_id = row[0]
			user_id = row[3]
			if user_id in self.by_user:
				self.by_user[user_id].add(item_id)
			else:
				self.by_user[user_id] = set([item_id])
			self.items.append(item_id)

		if show:
			for key, val in self.by_user.items():
				print(key, val)

		print('finished grouping')
		return self.by_user

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
		print('users:', len(self.by_user.items()))

	def alter_probs(self):
		#denominator
		denom = 0.0
		for prob in self.item_probs:
			denom += prob**self.neg_sample_adjust

		#adjust probs
		self.item_probs = [(x**self.neg_sample_adjust)/denom for x in self.item_probs]

	def pairs_by_user(self, items):
		for outer_i in range(len(items)):
			left = items[outer_i]
			for inner_i in range(len(items)):
				right = items[inner_i]
				self.pairs.append([left, right, 1]) #true pair
				self.neg_pair(left, items) #neg_pairs

			#if outer_i%100==0:
				#print('items done:', outer_i)

	def neg_pair(self, curr_item, neighbors):
		picked = []
		for _ in range(self.neg_sample_size):
			neg_item = ''
			dont_pick = ['', curr_item] + neighbors + picked

			while neg_item in dont_pick:
				neg_item = int(np.random.choice(self.items, 1, p=self.item_probs)[0])

			picked.append(neg_item)
			pair = [curr_item, neg_item, 0]
			self.pairs.append(pair)

	def get_pairs(self):
		n_users = 0
		for item_set in list(self.by_user.values()):
			#if self.drop_singles and len(item_set)<=2:
				#continue
			items = list(item_set)
			items.sort()
			self.pairs_by_user(items)
			n_users += 1
			print('{} users done'.format(n_users))

	def dump_pairs(self):
		self.n_pairs_inserted = 0
		cursor = self.conn.cursor()
		for row in self.pairs:
			cols = ['item_id', 'context_id', 'label']
			if utils.insert_row(cursor, self.pairs_tname, row, cols=cols):
				self.n_pairs_inserted += 1

		self.conn.commit()
		cursor.close()







if __name__ == '__main__':
	pass















