import utils
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class recommender():

	def __init__(self, name, data_dir, db_name):
		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		self.pairs_tname = '{}_pairs_train'.format(name)
		self.items_tname = '{}_items_train'.format(name)

	def load_all_emb(self):
		cursor = self.conn.cursor()
		utils.read_table(cursor, self.items_tname, cols=['id', 'embedding'])
		#utils.read_table(cursor, self.items_tname, cols=['title', 'embedding']) #make sqlite autoincrement starts at 0
		self.item_ids = []
		self.embs = []
		for row in cursor:
			if row[1]:
				self.item_ids.append(row[0])
				emb = [float(x) for x in row[1].split(',')]
				self.embs.append(emb)

		self.embs = np.array(self.embs)
		cursor.close()

	def get_emb(self, item_id):
		cursor = self.conn.cursor()
		sql_ = 'SELECT embedding FROM {} WHERE id=?;'.format(self.items_tname)
		cursor.execute(sql_, [item_id])
		result = cursor.fetchall()[0][0]
		cursor.close()
		return np.array([float(x) for x in result.split(',')])

	def simple_nearest(self, ref_vec, top_k):
		nearests = []
		cursor = self.conn.cursor()
		utils.read_table(cursor, self.items_tname, cols=['id', 'embedding'])
		#utils.read_table(cursor, self.items_tname, cols=['title', 'embedding'])
		for row in cursor:
			row_id = row[0]
			if row[1]:
				emb = np.array([float(x) for x in row[1].split(',')])
				d = np.linalg.norm(ref_vec - emb)
				data_point = (row_id, d)
				nearests.append(data_point)

		cursor.close()
		return sorted(nearests, key=lambda kv: float(kv[1]), reverse=False)[:top_k]

	def tsne_plot(self):
		self.load_all_emb()
		x_emb = TSNE(n_components=2).fit_transform(self.embs)
		fig, ax=plt.subplots(figsize=(3,3), dpi=300)
		x_min = 0.0
		x_max = 0.0
		y_min = 0.0
		y_max = 0.0
		for emb, word in zip(x_emb, self.item_ids):
			print('{}: {}, {}'.format(word, emb[0], emb[1]))
			ax.text(emb[0], emb[1], word)

			#set graph limits
			if emb[0] < x_min:
				x_min = emb[0]
			if emb[0] > x_max:
				x_max = emb[0]
			if emb[1] < y_min:
				y_min = emb[1]
			if emb[1] > y_max:
				y_max = emb[1]

		plt.ylim(y_min-abs(y_min*0.05), y_max+abs(y_max*0.05))
		plt.xlim(x_min-abs(x_min*0.05), x_max+abs(x_max*0.05))
		plt.show()








































