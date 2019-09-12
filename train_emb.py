import utils
import nets
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sqlite3


class emb_trainer():

	def __init__(self,
				 alpha,
				 db_name,
				 batch_size,
				 n_epoch,
				 emb_dim,
				 pairs_tname,
				 items_tname,
				 dump_id,
				 n_batch=-1,
				 name=None,
				 net_path=None,
				 early_stop=True
				 ):
		#db
		self.conn = sqlite3.connect(os.path.join('storage', db_name))
		self.pairs_tname = pairs_tname
		self.items_tname = items_tname
		self.dump_id = dump_id

		#io
		self.model_dir = 'bin'
		self.data_dir = 'storage'
		self.model_path = os.path.join('bin', '{}_{}_net.pickle'.format(name, items_tname))
		self.meta_path = os.path.join('storage', '{}_{}_meta.json'.format(name, items_tname))

		#init net
		self.emb_dim = emb_dim
		if name:
			self.init_new_model(name)
		else:
			self.init_exist_model(net_path)

		#loss & train
		self.criterion = nn.BCELoss()
		self.learning_rate = alpha
		self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
		self.batch_size = batch_size
		self.n_epoch = n_epoch
		self.n_batch = n_batch

		#early stop
		self.early_stop = early_stop
		self.epoch_losses = []
		self.non_decrease_streaks = 0
		self.non_decrease_thres = 50
		self.early_stop_thres = 0.00001 #percentage of epoch loss decrease

		

	def init_new_model(self, name):
		#merge buffer data with training data
		cursor = self.conn.cursor()
		buffer_pairs_tname = self.pairs_tname.split('_')
		buffer_pairs_tname[-1] = 'buffer'
		buffer_pairs_tname = '_'.join(buffer_pairs_tname)
		buffer_items_tname = self.items_tname.split('_')
		buffer_items_tname[-1] = 'buffer'
		buffer_items_tname = '_'.join(buffer_items_tname)
		utils.move_data(cursor, buffer_pairs_tname, self.pairs_tname)
		utils.move_data(cursor, buffer_items_tname, self.items_tname)
		self.conn.commit()
		cursor.close()

		#get item list & n unique items, cache item list into json
		cursor = self.conn.cursor()
		utils.read_table(cursor, self.items_tname, cols=['id'])
		self.item_ids = []
		for row in cursor:
			self.item_ids.append(row[0])
		cursor.close()

		#get u_unique_items
		self.n_unique_items = len(self.item_ids)
		meta = {'n_unique_items':self.n_unique_items}
		utils.json_dump(self.meta_path, meta)

		#init net
		self.net = nets.skip_gram(self.n_unique_items, name=name, emb_dim=self.emb_dim)
		print('Initiated new model')

	def init_exist_model(self, net_path):
		#init net
		self.net = utils.pickle_load(net_path)
		self.name = self.net.name
		self.meta_path = os.path.join(self.data_dir, '{}_meta.json'.format(self.name))
		self.pairs_tname = '{}_pairs_train'.format(self.name)
		self.items_tname = '{}_items_train'.format(self.name)

		#get item list
		self.n_unique_items = utils.json_load(self.meta_path)['n_unique_items']
		self.item_ids = list(range(self.n_unique_items))
		print('Loaded existing model')
		
	def test_pass(self):
		self.batch_size = 99
		self.n_batch = 1
		self.single_epoch()

	def load_batch(self):
		cursor = self.conn.cursor()
		cols = ['ref_id', 'context_id', 'label'] #context is also item
		utils.read_table(cursor, self.pairs_tname, cols=cols)
		batch = [1]
		while batch:
			batch = cursor.fetchmany(size=self.batch_size)
			yield batch
		cursor.close()

	def get_onehot(self, index_batch):
		vec_batch = []
		for index in index_batch:
			vec = [0 for x in range(self.n_unique_items)]
			vec[index] = 1
			vec_batch.append(vec)
		vec_batch = torch.tensor(vec_batch, dtype=torch.float)
		return vec_batch

	def batch2loss(self, batch):
		#forward pass
		item_ids = [item_id for item_id, context_id, label in batch if label==1]
		item_ids = torch.tensor(item_ids, dtype=torch.long)
		output = self.net(item_ids)
		context_vecs = np.array(output.detach())

		#get neg sampled ground truth
		con_vec_i = -1
		for item_id, context_id, label in batch:
			if label==1:
				con_vec_i += 1
			if con_vec_i!=-1:
				context_vecs[con_vec_i][context_id] = label

		#get loss
		context_vecs = torch.tensor(context_vecs, dtype=torch.float)
		loss = self.criterion(output, context_vecs)

		#print('init item id:', item_ids)
		#print('batch:', batch)
		#print('context:', context_vecs)
		#print('context shape:', context_vecs.shape)
		#print('loss:', loss)

		return loss

	def single_epoch(self, i_epoch=0):
		i_batch = 0
		running_loss = 0.0
		for batch in self.load_batch():
			if self.n_batch != -1 and i_batch >= self.n_batch:
				break

			if len(batch)==0:
				continue

			self.optimizer.zero_grad()
			loss = self.batch2loss(batch)
			running_loss += float(loss)
			#print('epoch: {}, batch: {}, loss: {}'.format(str(i_epoch+1), str(i_batch+1), str(float(loss.data))))
			loss.backward()
			self.optimizer.step()
			i_batch += 1

			#break

		epoch_loss = running_loss / float(i_batch)
		print('epoch: {}, loss: {}'.format(str(i_epoch+1), str(epoch_loss)))
		self.epoch_losses.append(epoch_loss)

	def train_loop(self):
		#try:
		for i_epoch in range(self.n_epoch):

			#early stop if loss does not decrease for certain epoches
			if self.non_decrease_streaks > self.non_decrease_thres and self.early_stop:
				print('early stopping')
				break

			#train one epoch
			self.single_epoch(i_epoch)

			#if loss doesn't drop enought amt comparing to last epoch, add 1 to streak
			if len(self.epoch_losses) >= 2:
				dec_pct = (self.epoch_losses[-2] - self.epoch_losses[-1]) / self.epoch_losses[-2]
				if dec_pct < self.early_stop_thres:
					self.non_decrease_streaks += 1

		#except:
			#pass

		print('Done training')
		utils.pickle_dump(self.model_path, self.net)
		print('pickled net')
		self.dump_emb()
	
	def dump_emb(self):
		cursor = self.conn.cursor()
		for item_id, embedding in zip(self.item_ids, self.net.embeddings.weight.detach()):
			emb_insert = [','.join([str(x) for x in list(embedding.numpy())])]
			utils.update_table(cursor, emb_insert, self.dump_id, item_id, 'embedding', self.items_tname)
		self.conn.commit()
		print('Saved embeddings')
		cursor.close()
		self.conn.close()


class user_rep():
	#deprecated

	def __init__(self, name, db_name, data_dir):
		#io
		self.data_dir = data_dir
		self.fav_fname = '{}_fav_lookup.csv'.format(name)

		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		self.items_tname = '{}_items_train'.format(name)
		self.users_tname = '{}_users_train'.format(name)

		#merge user buffer data with training data
		cursor = self.conn.cursor()
		buffer_users_tname = '{}_users_buffer'.format(name)
		utils.move_data(cursor, buffer_users_tname, self.users_tname)
		self.conn.commit()
		cursor.close()

	def loop_groups(self):
		fav_path = os.path.join(self.data_dir, self.fav_fname)
		with open(fav_path, 'r') as f:
			i = 0
			for line in f:
				row = line[:-1].split(',')
				i+=1
				self.single_user(row, i)

	def single_user(self, row, i): #user id is the first item of a row
		#check if user has favs
		user_web_id = row[0]
		if len(row) == 1:
			print('{}th user({}) has no favorites'.format(i, user_web_id))
			return

		#get embedding
		print('getting rep for {}th user: {}'.format(i, user_web_id))
		emb_arr = []
		for item_db_id in row[1:]:
			emb = utils.get_emb(self.conn, self.items_tname, item_db_id)
			emb_arr.append(emb)

		emb_arr = np.array(emb_arr)
		user_rep = np.mean(emb_arr, axis=0)
		user_rep = [','.join([str(x) for x in user_rep])]

		#insert embedding
		cursor = self.conn.cursor()
		utils.update_table(cursor, user_rep, 'web_id', user_web_id, 'embedding', self.users_tname)
		self.conn.commit()
		cursor.close()






if __name__ == '__main__':
	name = 'cla_00'
	db_name = 'cla_00.db'
	data_dir = 'storage'
	rep = user_rep(name, db_name, data_dir)
	rep.loop_groups()
























