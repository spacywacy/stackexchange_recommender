import utils
import requests
import json
import pickle
import os
import sqlite3
import numpy as np
from time import sleep
from hashlib import md5


class stack_api_wrapper():

	def __init__(self, name, db_name, data_dir, n_user_pages, user_pagesize):
		#io
		self.key_loc = 'resources/key'
		self.result_dir = data_dir
		self.user_ids_fname = '{}_user_ids.pickle'.format(name)
		self.item_ids_fname = '{}_item_ids.pickle'.format(name)
		self.groups_fname = '{}_groups.csv'.format(name)
		self.fav_lookup_fname = '{}_fav_lookup.csv'.format(name)
		self.new_user_favs_fname = '{}_new_user_favs.csv'.format(name)
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

		#db
		self.conn = sqlite3.connect(os.path.join(self.result_dir, db_name))
		self.user_tname = '{}_users_buffer'.format(name)
		self.item_tname = '{}_items_buffer'.format(name)
		self.item_train_tname = '{}_items_train'.format(name)

		#counters
		self.n_users_inserted = 0
		self.n_items_inserted = 0


	def api_call(self):
		self.get_users()
		self.ref_items_user()
		self.context_items()
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

	def ref_items_raw(self):
		#request header
		url = 'https://api.stackexchange.com/2.2/questions'
		params = {
			'order':'desc',
			'sort':'reputation',
			'site':self.site,
			'pagesize':self.user_pagesize,
			'page':'',
			'key':self.key
		}

		#request loop
		item_ids = []
		for i in range(1, self.n_user_pages+1):
			print('page:', i)
			params['page'] = i

			#get data
			json_items = utils.call_api(url, params)
			item_ids_in_loop = [x['question_id'] for x in json_items]

			#append data
			item_ids += item_ids_in_loop
			self.n_users_inserted += utils.store_questions(json_items, self.item_tname, self.conn)

			#delay
			sleep(self.delay)

		#cache user id list
		ids_path = os.path.join(self.result_dir, self.item_ids_fname)
		utils.pickle_dump(ids_path, item_ids)

	def ref_items_user(self):
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
		item_web_ids = []
		item_db_ids = []
		fav_path = os.path.join(self.result_dir, self.fav_lookup_fname)
		#how to preserve previous user fav information when new users are added?
		with open(fav_path, 'w') as f:
			for user_id in user_ids:
				print('get favs for {}th user'.format(i_user))
				url = 'https://api.stackexchange.com/2.2/users/{}/favorites'.format(user_id)

				#get & store data to db
				json_items = utils.call_api(url, params)
				item_web_ids += [x['question_id'] for x in json_items]
				tmp_db_ids, n_items = utils.store_questions(json_items,
															self.item_tname,
															self.conn,
															return_items=True)
				self.n_items_inserted += n_items
				item_db_ids += tmp_db_ids

				#write fav lookup to file
				line = ','.join([str(x) for x in [user_id]+tmp_db_ids]) + '\n'
				f.write(line)

				i_user += 1

				#delay
				sleep(self.delay)

		#cache user id list
		ids_path = os.path.join(self.result_dir, self.item_ids_fname)
		utils.pickle_dump(ids_path, list(zip(item_web_ids, item_db_ids)))

	def context_items(self):
		#load item ids
		pickle_path = os.path.join(self.result_dir, self.item_ids_fname)
		item_ids = utils.pickle_load(pickle_path)

		#request header
		params = {
			'order':'desc',
			'sort':'votes',
			'site':self.site,
			'pagesize':self.user_pagesize,
			'page':'',
			'key':self.key
		}

		#requestion loop
		i = 1
		groups_path = os.path.join(self.result_dir, self.groups_fname)
		with open(groups_path, 'w') as f:
			for item_id in item_ids:
				web_id = item_id[0]
				db_id = item_id[1]
				print('get related item for {}th item'.format(i))
				url = 'https://api.stackexchange.com/2.2/questions/{}/related'.format(web_id)

				#get & store data
				json_items = utils.call_api(url, params)
				stored = utils.store_questions(json_items,
												self.item_tname,
												self.conn,
												group_id=db_id,
												return_items=True)
				self.n_items_inserted += stored[1]
				f.write(','.join([str(x) for x in [db_id]+stored[0]]) + '\n')
				i += 1

				#delay
				sleep(self.delay)

	def get_single_user(self, user_web_id):
		url = 'https://api.stackexchange.com/2.2/users/{}'.format(user_web_id)
		params = {
			'order':'desc',
			'sort':'reputation',
			'site':self.site,
			'page':'',
			'key':self.key
		}
		json_items = utils.call_api(url, params)
		self.n_users_inserted += utils.store_users(json_items, self.user_tname, self.conn)

	def new_user_favs(self, user_web_id):
		#request header
		params = {
			'order':'desc',
			'sort':'votes',
			'site':self.site,
			'key':self.key
		}

		#data
		items_in_sys = []
		new_items = []

		#request for user favs
		print('getting favs for user: {}'.format(user_web_id))
		url = 'https://api.stackexchange.com/2.2/users/{}/favorites'.format(user_web_id)
		json_items = utils.call_api(url, params)
		
		#request & store user info
		self.get_single_user(user_web_id)
		
		#check if item in system
		item_web_ids = [x['question_id'] for x in json_items]
		for item_web_id in item_web_ids:
			cursor = self.conn.cursor()
			hash_val = md5(str(item_web_id).encode()).hexdigest()
			item_db_id = utils.check_row(cursor, self.item_train_tname, 'hash_val', hash_val)

			#check if item in system
			if item_db_id or item_db_id==0:
				items_in_sys.append(item_db_id)
			else:
				new_items.append(item_web_id)
			cursor.close()

		#update favs for the user
		fav_path = os.path.join(self.result_dir, self.fav_lookup_fname)
		utils.update_favs_csv(fav_path, user_web_id, items_in_sys)

		#dump new items
		#to do: insert new item into buffer table
		#new_fav_path = os.path.join(self.result_dir, self.new_user_favs_fname)
		#with open(new_fav_path, 'a+') as f:
			#for item_web_id in new_items:
				#f.write(str(item_web_id) + '\n')

		#delay
		sleep(self.delay)


class emb_pair_builder():

	def __init__(self, name, data_dir, db_name, pairs_tname, groups_fname, neg_sample_size):
		#db & io
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		#self.pairs_tname = '{}_itempairs_buffer'.format(name)
		self.pairs_tname = pairs_tname
		#self.groups_fname = '{}_groups.csv'.format(name)
		self.groups_fname = groups_fname
		self.result_dir = data_dir

		#cache
		self.pairs = [] #item_id, item_id, label
		self.groups = [] #[[item_id]]
		self.items = [] #[[item_id]]
		self.n_unique_items = 0
		self.n_items = 0

		#others
		self.neg_sample_size = neg_sample_size
		self.neg_sample_adjust = .75
		self.drop_singles = False
		

	def create_pairs(self):
		self.load_groups()
		self.counts_and_probs()
		self.get_pairs()
		self.dump_pairs()
		self.conn.close()
		print('Total of {} items, {} pairs'.format(len(self.items), len(self.pairs)))
		print('Inserted {} pairs'.format(self.n_pairs_inserted))

	
	def load_groups(self):
		group_path = os.path.join(self.result_dir, self.groups_fname)
		with open(group_path, 'r') as f:
			for line in f:
				row = line[:-1].split(',')
				self.groups.append(row)
				self.items += row

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

	def alter_probs(self):
		#denominator
		denom = 0.0
		for prob in self.item_probs:
			denom += prob**self.neg_sample_adjust

		#adjust probs
		self.item_probs = [(x**self.neg_sample_adjust)/denom for x in self.item_probs]

	def pairs_by_group(self, items):
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
		for item_set in self.groups:
			#if self.drop_singles and len(item_set)<=2:
				#continue
			items = list(item_set)
			items.sort()
			self.pairs_by_group(items)
			n_users += 1
			print('{} groups done'.format(n_users))

	def dump_pairs(self):
		self.n_pairs_inserted = 0
		cursor = self.conn.cursor()
		for row in self.pairs:
			cols = ['ref_id', 'context_id', 'label']
			if utils.insert_row(cursor, self.pairs_tname, row, cols=cols)[1]:
				self.n_pairs_inserted += 1

		self.conn.commit()
		cursor.close()


































