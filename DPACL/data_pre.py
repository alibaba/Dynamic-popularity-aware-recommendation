#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

df_i = pd.read_csv('data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv', low_memory=False)
df_u = pd.read_csv('data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv', low_memory=False)

df_u_view = df_u[df_u.behavior_type == 1]
df_u_view = df_u_view[["user_id","item_id","time"]]
df_u_view["time"] = pd.to_datetime(df_u_view.time)
df_u_view["user_id"] = pd.Categorical(df_u_view["user_id"])
df_u_view["item_id"] = pd.Categorical(df_u_view["item_id"])

item_num = df_u_view["item_id"].value_counts()
item_num = item_num[item_num >= 30]
item_set = set(item_num.index.tolist())
df_u_view = df_u_view[df_u_view.item_id.apply(lambda x: x in item_set)]

train_df = df_u_view[df_u_view.time < "2014-12-18 00:00:00"]
test_df = df_u_view[df_u_view.time >= "2014-12-18 00:00:00"]

users_d = defaultdict(lambda: len(users_d))
for user_id in np.sort(train_df.user_id.unique()):
    users_d[user_id]
train_df["user_no"] = train_df["user_id"].apply(lambda x : users_d[x])
items_d = defaultdict(lambda: len(items_d))
for item_id in np.sort(train_df.item_id.unique()):
    items_d[item_id]

train_df["item_no"] = train_df["item_id"].apply(lambda x : items_d[x])
train_df.to_pickle('data/train_df.pkl')
print("save train data to data/train_df.pkl")
print(train_df.nunique())

test_df["item_no"] = test_df["item_id"].apply(lambda x : items_d[x] if items_d[x] <= 82556 else -1)
test_df["user_no"] = test_df["user_id"].apply(lambda x : users_d[x] if users_d[x] <= 19834 else -1)
test_df = test_df[test_df["item_no"] != -1]
test_df = test_df[test_df["user_no"] != -1]
test_df.to_pickle('data/test_df.pkl')
print("save test data to data/test_df.pkl")
print(test_df.nunique())



