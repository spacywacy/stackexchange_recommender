import os
import sqlite3
from utils import create_table
from utils import insert_row


def init_tables(name, data_dir, db_name):
	tnames = [
		'{}_users_train'.format(name),
		'{}_users_buffer'.format(name),
		'{}_userpairs_train'.format(name),
		'{}_userpairs_buffer'.format(name),
		'{}_items_train'.format(name),
		'{}_items_buffer'.format(name),
		'{}_itempairs_train'.format(name),
		'{}_itempairs_buffer'.format(name),
		'{}_classify_train'.format(name),
		'{}_classify_test'.format(name)
	]

	#set_autos = [False, True, True, True, True, True]

	sql_snippets = [
	 	#users_train
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		name varchar(128),
		reputation integer,
		accept_rate integer,
		embedding varchar(512),
		link varchar(512),
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		CREATE INDEX {}_web_id ON {} (web_id);
		'''.format(tnames[0], tnames[0], tnames[0]),

		#user_buffer
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		name varchar(128),
		reputation integer,
		accept_rate integer,
		embedding varchar(512),
		link varchar(512),
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		CREATE INDEX {}_web_id ON {} (web_id);
		'''.format(tnames[1], tnames[1], tnames[1]),

		#userpairs_train
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		ref_id integer,
		context_id integer,
		label integer
		);
		'''.format(tnames[2]), #pairs_train_tname

		#userpairs_buffer
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		ref_id integer,
		context_id integer,
		label integer
		);
		'''.format(tnames[3]), #pairs_buffer_tname

		#items_train
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		title varchar(512),
		group_id integer, /*user or item web id*/
		embedding varchar(512),
		tags varchar(512),
		view_count integer,
		answer_count integer,
		score integer,
		link varchar(512),
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		'''.format(tnames[4], tnames[4], tnames[4]),

		#items_buffer
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		title varchar(512),
		group_id integer, /*user or item web id*/
		embedding varchar(512), 
		tags varchar(512),
		view_count integer,
		answer_count integer,
		score integer,
		link varchar(512),
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		'''.format(tnames[5], tnames[5], tnames[5]),

		#itempairs_train
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		ref_id integer,
		context_id integer,
		label integer
		);
		'''.format(tnames[6]),

		#itempairs_buffer
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		ref_id integer,
		context_id integer,
		label integer
		);
		'''.format(tnames[7]),

		#classify_train
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		user_rep varchar(512),
		user_meta varchar(512),
		item_rep varchar(512),
		item_meta varchar(512),
		label integer
		);
		'''.format(tnames[8]),

		#classify_test
		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		user_rep varchar(512),
		user_meta varchar(512),
		item_rep varchar(512),
		item_meta varchar(512),
		label integer
		);
		'''.format(tnames[9]),
	]

	with sqlite3.connect(os.path.join(data_dir, db_name)) as conn:
		for tname, sql_snippet in zip(tnames, sql_snippets):
			create_table(conn, tname, sql_snippet, auto_at_0=True)













