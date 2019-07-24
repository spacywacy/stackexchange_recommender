import os
import utils
import sqlite3
from build_dataset import stack_api_wrapper
from build_dataset import pair_builder


name = 'same_user_00'
data_dir = 'storage'
db_name = 'stack_data_00.db'
conn = sqlite3.connect(os.path.join(data_dir, db_name))
item_tname = '{}_items_train'.format(name)
raw_items = 'raw_items'

def init_():
	sql_ = '''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		title varchar(512),
		belong_to integer,
		embedding varchar(512),
		tags varchar(512),
		view_count integer,
		answer_count integer,
		score integer,
		hash_val char(32)
		);

		CREATE INDEX {}_hash_ ON {} (hash_val);
		'''.format(item_tname, item_tname, item_tname) #item_buffer_tname

	utils.create_table(conn, item_tname, sql_, auto_at_0=True)

def process_items():
	#get counts
	cursor = conn.cursor()	
	utils.read_table(cursor, raw_items, cols=['belong_to'])
	by_user = {}
	for row in cursor:
		belong_to = row[0]
		by_user[belong_to] = by_user.get(belong_to, 0) + 1
	cursor.close()

	#move data
	count_thres = 2
	cursor = conn.cursor()
	cols = utils.get_cols(cursor, raw_items)
	if 'id' in cols:
		cols.remove('id')
	utils.read_table(cursor, raw_items, cols=cols)
	for row in cursor:
		count = by_user[row[2]]
		if count > count_thres:
			cur_insert = conn.cursor()
			if utils.insert_row(cur_insert, item_tname, row, cols=cols):
				print('insert')
			cur_insert.close()

	conn.commit()
	cursor.close()



init_()
process_items()
conn.close()





















