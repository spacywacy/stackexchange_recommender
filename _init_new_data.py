import os
import utils
import sqlite3
from create_tables import init_tables
from build_dataset import stack_api_wrapper
from build_dataset import pair_builder
from train import emb_trainer
from recommend import recommender
import nets
from time import time
from create_tables import init_tables

#db & io
#name = 'fav_by'
#db_name = 'fav_by.db'
#name = 'fav_test'
#db_name = 'fav_test'
name = 'asked_by_test'
db_name = 'asked_by_test'


data_dir = 'storage'
fig_dir = 'graph'
model_dir = 'bin'
conn = sqlite3.connect(os.path.join(data_dir, db_name))

'''
#init tables
init_tables(name, data_dir, db_name)

#api calls
n_user_pages = 3
user_pagesize = 10
#api_tag = 'favorites'
api_tag = 'questions'
api_wrapper = stack_api_wrapper(name, db_name, data_dir, n_user_pages, user_pagesize, api_tag)
api_wrapper.api_call()


#build pairs
p_builder = pair_builder(name, data_dir, db_name, neg_sample_size=6)
p_builder.create_pairs()

'''

#train parameters
alpha = 0.01
batch_size = -1
n_epoch = 1000

#train
#trainer = emb_trainer(alpha,data_dir,fig_dir,model_dir,db_name,batch_size,n_epoch,name=name,early_stop=False)
#t0 = time()
#trainer.train_loop()
#t1 = time()

#verify
top_k = 30
rec = recommender(name, data_dir, db_name, top_k)
#rec.get_truth()
#score = rec.verify()
#print('score:', score)
#print('score: {}, time: {}'.format(score, t1-t0))
#'''

ref_vec = rec.
rec.simple_nearest(ref_vec, top_k)























