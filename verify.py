import utils
import os
import sqlite3
import numpy as np


class verify():

	def __init__(self, name, data_dir, db_name, top_k, k_embs=float('inf')):
		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		self.pairs_tname = '{}_pairs_train'.format(name)
		self.items_tname = '{}_items_train'.format(name)

		#io
		self.data_dir = 'storage'
		self.truth_path = '{}_truth.pickle'.format(name)
		self.truth_path = os.path.join(self.data_dir, self.truth_path)

		self.top_k = top_k
		self.k_embs = k_embs

	def 















