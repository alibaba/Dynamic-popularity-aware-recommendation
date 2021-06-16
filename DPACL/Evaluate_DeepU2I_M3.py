#!/usr/bin/env python
# coding: utf-8

import faiss
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

train_df = pd.read_pickle('data/train_df.pkl')
test_df = pd.read_pickle('data/test_df.pkl')
test_df = test_df[["item_no", "user_no", "time"]]
train_df = train_df[["item_no", "user_no", "time"]]

userid_vectors = np.load("userid_vectors_M5.npy", )
userid_index = np.load("userid_index_M5.npy")
itemid_vectors = np.load("itemid_vectors_M5.npy")
itemid_index = np.load("itemid_index_M5.npy")

item_df = train_df[train_df.time >= "2014-12-17 00:00:00"]
item_df = item_df.append(test_df)
userid_index_index = {}
for i in range(len(userid_index)):
    userid_index_index[userid_index[i]] = i
adjust_vector = {}
for i in range(24):
    end_time = datetime(2014, 12, 18) + timedelta(hours=i)
    start_time = end_time + timedelta(days=-1)
    last24hour_df = item_df[item_df.time >= start_time]
    last24hour_df = last24hour_df[last24hour_df.time < end_time]
    item_num_dict = last24hour_df["item_no"].value_counts().to_dict()
    adjust_vector[i] = np.array([np.log(item_num_dict.get(item_no, 0) + 1) / 20 for item_no in itemid_index])
recall_sum = {}
item_sum = {}
for k in [1, 5, 10, 50, 100, 300, 500, 1000]:
    recall_sum[k] = 0
    item_sum[k] = 0
userid_vectors_add = np.column_stack((userid_vectors, np.ones(userid_vectors.shape[0]))).astype(np.float32)

dim = 129
for i in range(24):
    index = faiss.IndexFlatIP(dim)
    itemid_vectors_add = np.column_stack((itemid_vectors, adjust_vector[i])).astype(np.float32)
    index.add(itemid_vectors_add)
    D, I = index.search(userid_vectors_add, 1000)
    start_time = datetime(2014, 12, 18) + timedelta(hours=i)
    end_time = start_time + timedelta(hours=1)
    test_df_hour = test_df[test_df.time >= start_time]
    test_df_hour = test_df_hour[test_df_hour.time < end_time]
    test_df_hour_userno_group = test_df_hour.groupby("user_no")
    test_userno_list = test_df_hour.user_no.unique().tolist()
    for j in range(len(test_userno_list)):
        userno = test_userno_list[j]
        userno_index = userid_index_index[userno]
        user_item_label = test_df_hour_userno_group.get_group(userno).item_no.tolist()
        for k in [1, 5, 10, 50, 100, 300, 500, 1000]:
            user_item_topK = set(itemid_index[I[userno_index][:k]].tolist())
            count = 0
            for item_no in user_item_label:
                if item_no in user_item_topK:
                    count += 1
            recall_sum[k] += count
            item_sum[k] += len(user_item_label)
    print("cal hour:", i)

for k in [1, 5, 10, 50, 100, 300, 500, 1000]:
    recall = recall_sum[k] / item_sum[k]
    print("Recall@" + str(k) + " = " + str(round(recall, 4)))
