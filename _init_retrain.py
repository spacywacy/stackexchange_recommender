import os
import sys
import utils
import sqlite3
from create_tables import init_tables
from train_emb import emb_trainer
from train_emb import user_rep
from recommend import recommender
import nets
from time import time
from create_tables import init_tables
from new_build_dataset import stack_api_wrapper
from new_build_dataset import emb_pair_builder
from classification import classifier
from nearest import near


#db & io
name = 'cla_021'
db_name = 'cla_021.db'
data_dir = 'storage'
fig_dir = 'graph'
model_dir = 'bin'
conn = sqlite3.connect(os.path.join(data_dir, db_name))

#api calls
n_pages = 2
pagesize = 2
api_wrapper = stack_api_wrapper(name, db_name, data_dir, n_pages, pagesize)
api_wrapper.ref_items_raw(sys.argv[1], sys.argv[2])
api_wrapper.context_items()
api_wrapper.conn.close()

#build item pairs
item_pairs_tname = '{}_itempairs_buffer'.format(name)
groups_fname = '{}_groups.csv'.format(name)
builder = emb_pair_builder(name, data_dir, db_name, item_pairs_tname, groups_fname, neg_sample_size=6)
builder.create_pairs()


#train item embeddings
alpha = 0.01
batch_size = 2000
n_epoch = 50
emb_dim = 30
pairs_tname = '{}_itempairs_train'.format(name)
items_tname = '{}_items_train'.format(name)
dump_id = 'id'
trainer = emb_trainer(alpha,db_name,batch_size,n_epoch,emb_dim,pairs_tname,items_tname,dump_id,name=name,early_stop=True)
t0 = time()
trainer.train_loop()
t1 = time()
print('train time:', t1-t0)


#classification
cla_neg_size = 6
emb_dim = 30
cla = classifier(name, data_dir, db_name, cla_neg_size, emb_dim)
cla.build_dataset()
cla.build_classifier()
cla.prob_rank_by_user()









































