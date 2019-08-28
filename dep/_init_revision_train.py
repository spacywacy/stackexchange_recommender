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

#db & io
name = 'same_user_00'
data_dir = 'storage'
db_name = 'stack_data_00.db'
fig_dir = 'graph'
model_dir = 'bin'
conn = sqlite3.connect(os.path.join(data_dir, db_name))

#train parameters
alpha = 0.001
batch_size = -1
n_epoch = 5000

#train & verify
trainer = emb_trainer(alpha,data_dir,fig_dir,model_dir,db_name,batch_size,n_epoch,name=name)
t0 = time()
trainer.train_loop()
t1 = time()
rec = recommender(name, data_dir, db_name)
rec.get_truth()
score = rec.verify()
print('score: {}, time: {}'.format(score, t1-t0))



























