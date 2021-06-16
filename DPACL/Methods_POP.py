#!/usr/bin/env python
# coding: utf-8

import pandas as pd

train_df = pd.read_pickle('data/train_df.pkl')
item_df = train_df[["item_no"]]

test_df = pd.read_pickle('data/test_df.pkl')
test_df = test_df[["item_no","user_no","time"]]
test_df_userno_group = test_df.groupby("user_no")
test_user = test_df["user_no"].unique().tolist()

item_num = item_df["item_no"].value_counts()
popitem_list = item_num.head(100).index.tolist()

print("Methods POP:")
for k in [1,5,10,50,100]:
    recall_sum = 0
    item_sum = 0
    for i in range(len(test_user)):
        user_item_label = test_df_userno_group.get_group(test_user[i]).item_no.tolist()
        user_item_topK = set(popitem_list[:k])
        count = 0
        for item_no in user_item_label:
            if item_no in user_item_topK:
                count+=1
        recall_sum += count
        item_sum += len(user_item_label)
    recall = recall_sum/item_sum
    print("Recall@" + str(k) + " = " + str(round(recall,4)))
