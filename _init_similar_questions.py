import os
import sys
import utils
import sqlite3
import nets
from nearest import near


#db & io
name = 'cla_021'
db_name = 'cla_021.db'
data_dir = 'storage'
fig_dir = 'graph'
model_dir = 'bin'
conn = sqlite3.connect(os.path.join(data_dir, db_name))

#get nearest items
ref_item = int(sys.argv[1])
near_ = near(name, conn, db_name)
result = near_.similar_items(ref_item)
for item in result:
	print(item)







