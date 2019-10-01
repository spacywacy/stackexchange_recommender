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
t0 = time()

#update favs
#new_user_id = '250448'
new_user_id = sys.argv[1]
api_wrapper = stack_api_wrapper(name, db_name, data_dir, None, None)
api_wrapper.new_user_favs(new_user_id)

#retrain user emb
cla_neg_size = 6
emb_dim = 30
cla = classifier(name, data_dir, db_name, cla_neg_size, emb_dim)
cla.build_dataset()
cla.recommend(int(new_user_id), 20)
t1 = time()
print('time elapsed:', t1-t0)










































