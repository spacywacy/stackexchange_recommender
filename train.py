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
				 data_dir,
				 fig_dir,
				 model_dir,
				 db_name,
				 batch_size,
				 n_epoch,
				 n_batch=-1,
				 name=None,
				 net_path=None,
				 neg_sample=True,
				 early_stop=True,
				 start_at_1=False,
				 net=None,
				 pairs_tname=None,
				 items_tname=None
				 ):
		#others
		self.start_at_1 = start_at_1
		self.tmp_net = net

		#db
		self.conn = sqlite3.connect(os.path.join(data_dir, db_name))
		if pairs_tname:
			self.pairs_tname = pairs_tname
		else:
			self.pairs_tname = '{}_pairs_train'.format(name)
		if items_tname:
			self.items_tname = items_tname
		else:
			self.items_tname = '{}_items_train'.format(name)

		#io
		self.fig_dir = fig_dir
		self.model_dir = model_dir
		self.data_dir = data_dir
		self.fig_path = os.path.join(fig_dir, '{}.png'.format(name))
		self.model_path = os.path.join(model_dir, '{}_trainer.pickle'.format(name))
		self.meta_path = os.path.join(data_dir, '{}_meta.json'.format(name))

		#init net
		self.neg_sample = neg_sample
		if name:
			self.init_new_model(name, neg_sample)
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

		

	def init_new_model(self, name, neg_sample):
		#merge buffer data with training data
		cursor = self.conn.cursor()
		buffer_pairs_tname = '{}_pairs_buffer'.format(name)
		buffer_items_tname = '{}_items_buffer'.format(name)
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

		#adjust for starting point
		if self.start_at_1:
			self.item_ids = [x-1 for x in self.item_ids]

		self.n_unique_items = len(self.item_ids)
		meta = {'item_ids':self.item_ids}
		utils.json_dump(self.meta_path, meta)

		#init net
		if self.tmp_net:
			self.net = self.tmp_net(self.n_unique_items, name=name)
		else:
			if neg_sample:
				self.net = nets.bilinear(self.n_unique_items, name=name)
			else:
				self.net = nets.simple_emb(self.n_unique_items, name=name)


		print('Initiated new model')

	def init_exist_model(self, net_path):
		#init net
		self.net = utils.pickle_load(net_path)
		self.name = self.net.name
		self.meta_path = os.path.join(self.data_dir, '{}_meta.json'.format(self.name))
		self.pairs_tname = '{}_pairs_train'.format(self.name)
		self.items_tname = '{}_items_train'.format(self.name)
		if self.net.neg_sample:
			self.fig_path = os.path.join(self.fig_dir, '{}_neg.png'.format(self.name))
			self.model_path = os.path.join(self.model_dir, '{}_trainer_neg.pickle'.format(self.name))

		#get item list
		self.item_ids = utils.json_load(self.meta_path)['item_ids']
		self.n_unique_items = len(self.item_ids)

		print('Loaded existing model')
		
	def test_pass(self):
		self.batch_size = 99
		self.n_batch = 1
		self.single_epoch()

	def load_batch(self):
		cursor = self.conn.cursor()
		cols = ['item_id', 'context_id', 'label'] #context is also item
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
		#get index batches
		word_i_batch = [x[0] for x in batch]
		context_i_batch = [x[1] for x in batch]
		label = [x[2] for x in batch]

		#adjust for starting point
		if self.start_at_1:
			word_i_batch = [x-1 for x in word_i_batch]
			context_i_batch = [x-1 for x in context_i_batch]

		#print(batch, label)

		#do forward pass & get loss
		if self.neg_sample:
			word_i_batch = torch.tensor(word_i_batch, dtype=torch.long)
			context_i_batch = torch.tensor(context_i_batch, dtype=torch.long)
			label = torch.tensor(label, dtype=torch.float).unsqueeze(0).view(-1, 1)
			output = self.net(word_i_batch, context_i_batch)
			#print(float(output), float(label))
			loss = self.criterion(output, label)

		else:
			word_i_batch = torch.tensor(word_i_batch, dtype=torch.long)
			context_vec_batch = self.get_onehot(context_i_batch)
			output = self.net(word_i_batch)
			#print(word_i_batch, output, context_vec_batch)
			loss = self.criterion(output, context_vec_batch)
		
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
			print('epoch: {}, batch: {}, loss: {}'.format(str(i_epoch+1), str(i_batch+1), str(float(loss.data))))
			loss.backward()
			self.optimizer.step()
			i_batch += 1

		epoch_loss = running_loss / float(i_batch)
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
		for item_id, embedding in zip(self.item_ids, self.net.word_embeddings.weight.detach()):
			emb_insert = [','.join([str(x) for x in list(embedding.numpy())])]
			utils.update_table(cursor, emb_insert, 'id', item_id, 'embedding', self.items_tname)
		self.conn.commit()
		print('Saved embeddings')
		cursor.close()
		self.conn.close()









if __name__ == '__main__':
	pass
























