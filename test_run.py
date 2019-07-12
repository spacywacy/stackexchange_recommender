import os
import utils
import sqlite3
from create_tables import init_tables
from build_dataset import stack_api_wrapper
from build_dataset import pair_builder
from train import emb_trainer
from recommend import recommender


name = 'utils_tests'
data_dir = 'storage'
db_name = 'utils_tests.db'
fig_dir = 'graph'
model_dir = 'bin'
user_path = os.path.join(data_dir, 'mock_users.pickle')
item_path = os.path.join(data_dir, 'mock_ques.pickle')
conn = sqlite3.connect(os.path.join(data_dir, db_name))
user_tname = 'utils_tests_users'
item_tname = 'utils_tests_items_buffer'
pair_tname = 'utils_tests_pairs_buffer'
n_user_pages = 5
user_pagesize = 5
alpha = 0.001
batch_size = -1
n_epoch = 60000


#init_tables(name, data_dir, db_name)
#api_wrapper = stack_api_wrapper(name, db_name, data_dir, n_user_pages, user_pagesize)
#api_wrapper.api_call()
p_builder = pair_builder(name, data_dir, db_name, neg_sample_size=5)
#p_builder.create_pairs()


#trainer = emb_trainer(alpha,data_dir,fig_dir,model_dir,db_name,batch_size,n_epoch,name=name)
#trainer.train_loop()

p_builder.group_by_user(show=True, debug=True)
rec = recommender(name, data_dir, db_name)
for item_id in range(29):
	ref = rec.get_emb(item_id)
	result = rec.simple_nearest(ref, 6)
	result = [x[0] for x in result]
	print('{}, {}'.format(item_id, str(result)))










conn.close()
























