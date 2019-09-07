import os
import utils
import sqlite3
from create_tables import init_tables
from train_emb import item_emb_trainer
from train_emb import user_rep
from recommend import recommender
import nets
from time import time
from create_tables import init_tables
from new_build_dataset import stack_api_wrapper
from new_build_dataset import emb_pair_builder
from classification import classifier


#db & io
name = 'cla_01'
db_name = 'cla_01.db'
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
api_wrapper = stack_api_wrapper(name, db_name, data_dir, n_user_pages, user_pagesize)
api_wrapper.api_call()

#build pairs
builder = emb_pair_builder(name, data_dir, db_name, neg_sample_size=6)
builder.create_pairs()
'''

#train item embeddings
#alpha = 0.01
#batch_size = 2000
#n_epoch = 50
#emb_dim = 30
#trainer = item_emb_trainer(alpha,data_dir,fig_dir,model_dir,db_name,batch_size,n_epoch,emb_dim,name=name,early_stop=True)
#t0 = time()
#trainer.train_loop()
#t1 = time()
#print('train time:', t1-t0)

#verify item embeddings
#top_k = 30
#rec = recommender(name, data_dir, db_name, top_k, k_embs=1000)
#rec.get_truth_related()
#score = rec.verify()
#print('score:', score)


#get user representations
#user_rep_wrapper = user_rep(name, db_name, data_dir)
#user_rep_wrapper.loop_groups()

#classification
cla_neg_size = 6
emb_dim = 30
cla = classifier(name, data_dir, db_name, cla_neg_size, emb_dim)
#cla.build_dataset()
cla.load_data()
cla.build_classifier()
cla.evaluation()









































