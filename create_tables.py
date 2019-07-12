import os
import sqlite3
from utils import create_table
from utils import insert_row


def init_tables(name, data_dir, db_name):
	tnames = [
		'{}_users'.format(name),
		'{}_items_train'.format(name),
		'{}_items_buffer'.format(name),
		'{}_pairs_train'.format(name),
		'{}_pairs_buffer'.format(name)
	]

	set_autos = [False, True, True, True, True]

	sql_snippets = [
		'''
		CREATE TABLE {} (
		web_id integer primary key,
		name varchar(128),
		reputation integer,
		accept_rate integer,
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		'''.format(tnames[0], tnames[0], tnames[0]), #user_tname

		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		title varchar(512),
		belong_to integer,
		embedding varchar(512), /*not sure what dtype to use*/
		tags varchar(512),
		view_count integer,
		answer_count integer,
		score integer,
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		'''.format(tnames[1], tnames[1], tnames[1]), #item_tname_train

		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		web_id integer,
		title varchar(512),
		belong_to integer,
		embedding varchar(512), /*not sure what dtype to use*/
		tags varchar(512),
		view_count integer,
		answer_count integer,
		score integer,
		hash_val char(32)
		);

		CREATE INDEX {}_hash ON {} (hash_val);
		'''.format(tnames[2], tnames[2], tnames[2]), #item_buffer_tname

		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		item_id integer,
		context_id integer,
		label integer
		);
		'''.format(tnames[3]), #pairs_train_tname

		'''
		CREATE TABLE {} (
		id integer primary key autoincrement,
		item_id integer,
		context_id integer,
		label integer
		);
		'''.format(tnames[4]) #pairs_buffer_tname
	]

	with sqlite3.connect(os.path.join(data_dir, db_name)) as conn:
		for tname, sql_snippet, set_auto in zip(tnames, sql_snippets, set_autos):
			create_table(conn, tname, sql_snippet, auto_at_0=set_auto)













