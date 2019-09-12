import utils
import os
import sqlite3
import numpy as np
from sklearn import metrics


class near():

	def __init__(self, name, conn, db_name):
		#db
		self.conn = conn
		self.item_tname = '{}_items_train'.format(name)
		self.test_tname = '{}_classify_test'.format(name)

		#io
		self.data_dir = 'storage'
		self.fav_path = os.path.join(self.data_dir, '{}_fav_train.csv'.format(name))

		#data
		self.test_groups = {}
		self.train_groups = {}
		self.user_favs = [] #[user_web_id, [k_fav_top_item_ids]]
		self.top_k_lookup = {} #{user_web_id: set(item_db_ids)}

		#eval
		self.truth = []
		self.predicted = []

		#meta
		self.k_near = 10
		self.k_fav = 10

	def get_eval(self):
		self.load_favs()
		self.build_lookup()
		self.loop_test()
		self.evaluation()

	def load_favs(self):
		with open(self.fav_path, 'r') as f:
			for line in f:
				row = line[:-1].split(',')
				user_web_id = int(row[0])
				item_db_ids = [int(x) for x in row[1:]]
				self.user_favs.append([user_web_id, item_db_ids[:self.k_fav]])

	def build_lookup(self):
		for user_id, fav_items in self.user_favs:
			self.top_k_lookup[user_id] = self.user_top_k(user_id, fav_items)
		print('lookup built')

	def user_top_k(self, user_id, fav_items):
		print('getting top items for user:', user_id)
		rec_items = set()
		for item_id in fav_items:
			rec_items = rec_items.union(self.item_nearest(item_id))
		return rec_items

	def item_nearest(self, ref_item_id):
		ref_vec = utils.get_emb(self.conn, self.item_tname, ref_item_id)
		nearests = []
		cursor = self.conn.cursor()
		utils.read_table(cursor, self.item_tname, cols=['id', 'embedding'])
		for row in cursor:
			row_id = row[0]
			if row[1]:
				emb = np.array([float(x) for x in row[1].split(',')])
				d = np.linalg.norm(ref_vec - emb)
				data_point = (row_id, d)
				nearests.append(data_point)

		cursor.close()
		nearests = sorted(nearests, key=lambda kv: float(kv[1]), reverse=False)[:self.k_near]
		return set([x[0] for x in nearests])

	def predict(self, user_web_id, item_db_id):
		if item_db_id in self.top_k_lookup[user_web_id]:
			return 1
		else:
			return 0

	def loop_test(self):
		cursor = self.conn.cursor()
		cols = ['user_web_id', 'item_db_id', 'label']
		utils.read_table(cursor, self.test_tname, cols=cols)
		for row in cursor:
			self.truth.append(row[-1])
			user_web_id = row[0]
			item_db_id = row[1]
			self.predicted.append(self.predict(user_web_id, item_db_id))
		cursor.close()

	def evaluation(self):
		accuracy = metrics.accuracy_score(self.truth, self.predicted)
		recall = metrics.recall_score(self.truth, self.predicted)
		precision = metrics.precision_score(self.truth, self.predicted)
		print('accuracy:', accuracy)
		print('recall:', recall)
		print('precision:', precision)












































