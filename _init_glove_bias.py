import os
import utils
import sqlite3
from create_tables import init_tables
from build_dataset import stack_api_wrapper
from build_dataset import pair_builder
from train import emb_trainer
from recommend import recommender
import nets


name = 'same_user_00'
#net_path = 'bin/same_user_00_trainer.pickle'
data_dir = 'storage'
db_name = 'stack_data_00.db'
fig_dir = 'graph'
model_dir = 'bin'
conn = sqlite3.connect(os.path.join(data_dir, db_name))
user_tname = '{}_users'.format(name)
item_tname = '{}_items_buffer'.format(name)
pair_tname = '{}_pairs_buffer'.format(name)
#n_user_pages = 100
#user_pagesize = 90
alpha = 0.001
batch_size = -1
n_epoch = 5000
net = nets.bias_bilinear

p_builder = pair_builder(name, data_dir, db_name, neg_sample_size=6)
#p_builder.create_pairs()


#'''
#trainer = emb_trainer(alpha,data_dir,fig_dir,model_dir,db_name,batch_size,n_epoch,name=name,net=net)
#trainer = emb_trainer(alpha,data_dir,fig_dir,model_dir,db_name,batch_size,n_epoch,net_path=net_path)
#trainer.train_loop()


by_user_table = p_builder.group_by_user(show=False, debug=True)

def group_lookup():
	lookup = {}
	for key, item_list in by_user_table.items():
		for item in item_list:
			if item in lookup:
				lookup[item].append(item_list)
			else:
				lookup[item] = [item_list]

	table_path = 'storage/by_user_lookup.pickle'
	utils.pickle_dump(table_path, lookup)

#group_lookup()
by_user_lookup = utils.pickle_load('storage/by_user_lookup.pickle')
rec = recommender(name, data_dir, db_name)
for item_id in range(1000,1300):
	ref = rec.get_emb(item_id)
	result = rec.simple_nearest(ref, 6)
	result = [x[0] for x in result]
	grouped = str(by_user_lookup[item_id])
	print('{}, {}, {}'.format(item_id, str(result), grouped))
#'''












conn.close()
















