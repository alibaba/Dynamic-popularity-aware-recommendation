#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from datetime import datetime, timedelta

train_df = pd.read_pickle('data/train_df.pkl')
test_df = pd.read_pickle('data/test_df.pkl')
test_df = test_df[["item_no", "user_no", "time"]]
train_df = train_df[["item_no", "user_no", "time"]]
train_df = train_df[train_df.time >= "2014-12-17 00:00:00"]
train_df = train_df.append(test_df)

pop_realtime_list_hh = {}
for i in range(24):
    end_time = datetime(2014, 12, 18) + timedelta(hours=i)
    start_time = end_time + timedelta(days=-1)
    last24hour_df = train_df[train_df.time >= start_time]
    last24hour_df = last24hour_df[last24hour_df.time < end_time]
    item_num = last24hour_df["item_no"].value_counts()
    pop_realtime_list_hh[i] = item_num.head(100).index.tolist()

print("Methods_POP_RealTime:")
for k in [1, 5, 10, 50, 100]:
    recall_sum = 0
    item_sum = 0
    for i in range(24):
        start_time = datetime(2014, 12, 18) + timedelta(hours=i)
        end_time = start_time + timedelta(hours=1)
        test_df_hour = test_df[test_df.time >= start_time]
        test_df_hour = test_df_hour[test_df_hour.time < end_time]
        test_df_hour_userno_group = test_df_hour.groupby("user_no")
        test_userno_list = test_df_hour.user_no.unique().tolist()
        for j in range(len(test_userno_list)):
            user_item_label = test_df_hour_userno_group.get_group(test_userno_list[j]).item_no.tolist()
            user_item_topK = set(pop_realtime_list_hh[i][:k])
            count = 0
            for item_no in user_item_label:
                if item_no in user_item_topK:
                    count += 1
            recall_sum += count
            item_sum += len(user_item_label)
    recall = recall_sum / item_sum
    print("Recall@" + str(k) + " = " + str(round(recall, 4)))


