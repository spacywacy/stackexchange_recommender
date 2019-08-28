import requests
import json
import pickle
import os
import sqlite3
from hashlib import md5


def test():
	pass


#------------------basics----------------------------

def pickle_dump(fname, obj_):
	with open(fname, 'wb') as f:
		pickle.dump(obj_, f)

def pickle_load(fname):
	with open(fname, 'rb') as f:
		obj_ = pickle.load(f)
	return obj_

def json_dump(fname, obj_):
	with open(fname, 'w') as f:
		json.dump(obj_, f)

def json_load(fname):
	with open(fname, 'r') as f:
		obj_ = json.load(f)
	return obj_

def zero_padding(num, pad_to=6):
	n_pad = pad_to - len(str(num))
	if n_pad >=0:
		zeros = ''.join(['0' for x in range(n_pad)])
		return zeros + str(num)
	else:
		return str(num)


#------------------api&dataset-building----------------------------

def call_api(url, params, raw=False):
	res = requests.get(url, params=params)
	url_obj = res.content.decode('utf-8')
	if raw:
		return url_obj
	else:
		try:
			return json.loads(url_obj)['items']
		except:
			print(url_obj)

def store_users(json_items, tname, conn):
	cursor = conn.cursor()
	n_items = 0
	for user in json_items:
		web_id = user.get('user_id', None)
		name = user.get('display_name', None)
		reputation = user.get('reputation', None)
		accept_rate = user.get('accept_rate', None)
		row = [web_id, name, reputation, accept_rate]
		row_str = ''.join([str(x) for x in row])
		hash_val = md5(row_str.encode()).hexdigest()
		row += [hash_val]
		cols = ['web_id', 'name', 'reputation', 'accept_rate', 'hash_val']
		insert_row(cursor, tname, row, cols=cols, check_key='hash_val', check_key_val=hash_val)
		n_items+=1
	conn.commit()
	cursor.close()
	return n_items

def store_questions(json_items, fav_by, tname, conn):
	cursor = conn.cursor()
	n_items = 0
	for question in json_items:
		web_id = question.get('question_id', None)
		title = question.get('title', None)
		#belong_to = question.get('owner', {'user_id':None})['user_id']
		tags = ','.join(question.get('tags', []))
		view_count = question.get('view_count', None)
		answer_count = question.get('answer_count', None)
		row = [web_id, title, fav_by, tags, view_count, answer_count]
		row_str = ''.join([str(x) for x in row])
		hash_val = md5(row_str.encode()).hexdigest()
		row += [hash_val]
		cols = ['web_id', 'title', 'group_id', 'tags', 'view_count', 'answer_count', 'hash_val']
		insert_row(cursor, tname, row, cols=cols, check_key='hash_val', check_key_val=hash_val)
		n_items+=1
	conn.commit()
	cursor.close()
	return n_items

def store_questions_cache(json_items, tname, conn, count_thres=3):
	cursor = conn.cursor()
	n_items = 0
	rows = []
	count_by_user = {} #user: count
	cols = ['web_id', 'title', 'belong_to', 'tags', 'view_count', 'answer_count', 'hash_val']

	for question in json_items:
		web_id = question.get('question_id', None)
		title = question.get('title', None)
		belong_to = question.get('owner', {'user_id':None})['user_id']
		tags = ','.join(question.get('tags', []))
		view_count = question.get('view_count', None)
		answer_count = question.get('answer_count', None)
		row = [web_id, title, belong_to, tags, view_count, answer_count]
		row_str = ''.join([str(x) for x in row])
		hash_val = md5(row_str.encode()).hexdigest()
		row += [hash_val]
		rows.append(row)

		#get count by user
		count_by_user[belong_to] = count_by_user.get(belong_to, 0) + 1

	for row in rows:
		belong_to = row[2]
		if count_by_user[belong_to] >= count_thres:
			insert_row(cursor, tname, row, cols=cols, check_key='hash_val', check_key_val=hash_val)
			n_items+=1

	conn.commit()
	cursor.close()
	return n_items

def store_pairs(pairs, tname, conn):
	cursor = conn.cursor()
	for pair in pairs:
		item_id = pair[0]
		context_id = pair[1]
		label = pair[2]
		row = [item_id, context_id, label]
		cols = ['item_id', 'context_id', 'label']
		insert_row(cursor, tname, row, cols=cols)
	conn.commit()
	cursor.close()

#------------------database----------------------------

def insert_row(cursor, table_name, row, cols=None, check_key=None, check_key_val=None):
	if check_key:
		if check_row(cursor, table_name, check_key, check_key_val):
			print('Row already exists:', check_key_val)
			return False

	row_size = len(row)
	place_holders = ','.join(['?' for x in range(row_size)])
	if cols == None:
		sql_str = 'INSERT INTO {} VALUES ({});'.format(table_name, place_holders)
	else:
		col_str = ','.join(cols)
		sql_str = 'INSERT INTO {}({}) VALUES ({});'.format(table_name, col_str, place_holders)
	cursor.execute(sql_str, row)
	return True

def check_row(cursor, table_name, key_name, key_val):
	sql_ = 'SELECT EXISTS(SELECT 1 FROM {} WHERE {}=? LIMIT 1);'.format(table_name, key_name)
	cursor.execute(sql_, [key_val])
	z = cursor.fetchone()[0]
	if z == 1:
		return True
	else:
		return False

def check_table(conn, table_name):
	cursor = conn.cursor()
	sql_ = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND name = ?;'
	cursor.execute(sql_, [table_name])
	if_exist = len(list(cursor)) > 0 
	cursor.close()
	return if_exist

def create_table(conn, table_name, sql, auto_at_0=False):
	cur = conn.cursor()
	if not check_table(conn, table_name):
		cur.executescript(sql)
		if auto_at_0:
			set_auto_increment(cur, table_name)
		conn.commit()
		print('table: {} created'.format(table_name))
	else:
		print('table: {} already exists'.format(table_name))
	cur.close()


def read_table(cursor, table_name, cols=None, id_not_neg1=True):
	if cols:
		cols_str = ','.join(cols)
	else:
		cols_str = '*'

	if id_not_neg1:
		sql_ = '''
				SELECT {}
				FROM {}
				WHERE id != -1
				ORDER BY id;
			   '''.format(cols_str, table_name)
	else:
		sql_ = '''
				SELECT {}
				FROM {}
				ORDER BY id;
			   '''.format(cols_str, table_name)
	cursor.execute(sql_)

def update_table(cursor, row, key_col, key_val, cols, table_name):
	if type(cols)==list:
		set_clauses = ','.join(['{}=?'.format(x) for x in cols])
	else:
		set_clauses = '{}=?'.format(cols)
	where_clause = '{}=?'.format(key_col)
	sql_str = 'UPDATE {} SET {} WHERE {};'.format(table_name, set_clauses, where_clause)
	cursor.execute(sql_str, row+[key_val])
	#print('updated row')

def get_cols(cursor, table_name):
	sql_ = 'select * from {}'.format(table_name)
	cursor.execute(sql_)
	return list(map(lambda x: x[0], cursor.description))

def set_auto_increment(cursor, tname):
	#add id=-1 row back in
	mock_row = [-1]
	cols = ['id']
	insert_row(cursor, tname, mock_row, cols=cols)

	#set seq
	sql_ = 'UPDATE SQLITE_SEQUENCE SET seq=-1 WHERE name=\'{}\';'.format(tname)
	cursor.execute(sql_)

def move_data(cursor, from_tname, to_tname, delete_original=True):
	#move data
	cols = get_cols(cursor, from_tname)
	if 'id' in cols:
		cols.remove('id')
	cols_str = ','.join(cols)
	sql_ = '''
			INSERT INTO {} ({})
			SELECT {} FROM {}
			WHERE id != -1
			ORDER BY id;
		   '''.format(to_tname, cols_str, cols_str, from_tname)
	cursor.execute(sql_)

	#remove original data
	if delete_original:
		sql_ = 'DELETE FROM {};'.format(from_tname)
		cursor.execute(sql_)

	#reset auto increment to start at 0
	set_auto_increment(cursor, from_tname)

	


























if __name__ == '__main__':
	test()















