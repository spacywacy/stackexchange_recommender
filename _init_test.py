import os
import utils
import sqlite3
from classification import user_rep_emb

'''
def load_groups():
	groups = []
	with open('storage/cla_01_fav_train.csv', 'r') as f:
		for line in f:
			row = line[:-1].split(',')
			user_web_id = row[0]
			item_db_ids = [int(x) for x in row[1:]]
			if len(item_db_ids) == 0:
				continue
			groups.append([user_web_id, item_db_ids])

	return groups

groups = load_groups()
u_rep = user_rep_emb(groups, None, 'test')
u_rep.reverse_lookup()
'''


items = utils.pickle_load('storage/cla_01_item_ids.pickle')
print(len(items))

















