#!/usr/bin/env python
# coding: utf-8
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
from model_code.deep_u2i import model_fn

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_string("tables", "", "input tables")
tf.app.flags.DEFINE_integer("batch_size", 512, "batch_size")
tf.app.flags.DEFINE_integer("shuffle_size", 100000, "batch_size")
tf.app.flags.DEFINE_integer("eval_batch_size", 1000, "eval_batch_size")
tf.app.flags.DEFINE_string("phase", "train", "train or predict or delete")
tf.app.flags.DEFINE_string("predict_type", "output", "predict type")
tf.app.flags.DEFINE_float('lr', 0.001, 'learning_rate')
tf.app.flags.DEFINE_string('checkpointDir', 'check_point_dir', 'oss info')
tf.app.flags.DEFINE_string("model", "deep_matching_u2i", "model name")
tf.app.flags.DEFINE_string('outputs', 'predict_result', "predict saved name")
tf.app.flags.DEFINE_integer('train_steps', 5000, 'train steps')
tf.app.flags.DEFINE_integer('eval_steps', 10000, 'eval steps')
tf.app.flags.DEFINE_integer('neg_num', 511, 'number of negative sample')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer methods')
tf.app.flags.DEFINE_integer("embedding_size", 128, "embedding dim")
tf.app.flags.DEFINE_float('id_dropout_keep_prob', 0.0, 'dropout_keep_prob')
tf.app.flags.DEFINE_float('eval_interval_secs', 10000, 'eval_interval_secs')
tf.app.flags.DEFINE_float('user_vector_lambda', 0.0, 'user_vector_lambda')
tf.app.flags.DEFINE_integer('user_hash_num', 19835, 'user_hash_num')
tf.app.flags.DEFINE_integer('item_hash_num', 82557, 'item_hash_num')
tf.app.flags.DEFINE_string('predict_checkpoint', '', 'predict_checkpoint')
tf.app.flags.DEFINE_integer("task_index", None, "Worker task index")
tf.app.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.app.flags.DEFINE_string('f', '', 'kernel')

ModeKeys = {
    'train': tf.estimator.ModeKeys.TRAIN,
    'eval': tf.estimator.ModeKeys.EVAL,
    'predict': tf.estimator.ModeKeys.PREDICT
}

Optimizer = {
    'adam': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'adagrad': tf.train.AdagradOptimizer
}

FLAGS = tf.app.flags.FLAGS

train_df = pd.read_pickle('data/train_df.pkl')
test_df = pd.read_pickle('data/test_df.pkl')
train_df = train_df[["item_no", "user_no", "time"]]
test_df = test_df[["item_no", "user_no", "time"]]


def input_fn():
    def _parse(v1, v2):
        feature_cols = {}
        feature_cols['user_no'] = v1
        feature_cols['item_no'] = v2
        labels = tf.constant([0])
        return feature_cols, labels

    dataset = tf.data.Dataset.from_tensor_slices((train_df_tmp["user_no"], train_df_tmp["item_no"]))
    dataset = dataset.map(_parse)
    if FLAGS.phase == 'train':
        dataset = dataset.shuffle(FLAGS.shuffle_size).repeat().batch(FLAGS.batch_size)
    else:
        dataset = dataset.batch(FLAGS.batch_size)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn():
    def _parse(v1, v2):
        feature_cols = {}
        feature_cols['user_no'] = v1
        feature_cols['item_no'] = v2
        labels = tf.constant([0])
        return feature_cols, labels

    dataset = tf.data.Dataset.from_tensor_slices((test_df_tmp["user_no"], test_df_tmp["item_no"]))
    dataset = dataset.map(_parse)
    dataset = dataset.shuffle(100000).repeat().batch(FLAGS.eval_batch_size)

    return dataset.make_one_shot_iterator().get_next()


params = {
    'embedding_size': FLAGS.embedding_size,
    'learning_rate': FLAGS.lr,
    'optimizer': Optimizer[FLAGS.optimizer],
    'predict_type': "item",
    'batch_size': FLAGS.batch_size,
    'user_hash_num': FLAGS.user_hash_num,
    'item_hash_num': FLAGS.item_hash_num
}
model_name = "./model_m4/"
model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_name)

train_df_tmp = train_df.sample(frac=1)
epochs = 30
train_steps = int(len(train_df_tmp) / FLAGS.batch_size * epochs)
model.train(input_fn, steps=train_steps)
test_df_tmp = test_df
model.evaluate(eval_input_fn, steps=500)

# output item vector
model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_name)
item_dict = {}
item_dict["item_no"] = train_df.item_no.unique()
item_dict["user_no"] = np.zeros(len(item_dict["item_no"]))
test_df_tmp = pd.DataFrame(item_dict)
predictions = list(itertools.islice(model.predict(input_fn=eval_input_fn), 0, len(item_dict["item_no"]) + 2))
item_vec_map = {}
vec = set()
for i in range(len(predictions)):
    item_vec_map[predictions[i]["id"]] = predictions[i]["predict_result"]
    vec.add(predictions[i]["predict_result"])
id_index = []
id_vectors = []
for key in item_vec_map.keys():
    id_index.append(key)
    vector = np.array(item_vec_map[key].decode('utf8').split(',')).astype(np.float32)
    id_vectors.append(vector)
id_vectors = np.array(id_vectors)
id_index = np.array(id_index)
np.save("itemid_vectors_m4.npy", id_vectors)
np.save("itemid_index_m4.npy", id_index)
print("output user itemid")

# output user vector
params = {
    'embedding_size': FLAGS.embedding_size,
    'learning_rate': FLAGS.lr,
    'optimizer': Optimizer[FLAGS.optimizer],
    'neg_num': FLAGS.neg_num,
    'predict_type': "user",
    'batch_size': FLAGS.batch_size,
    'user_hash_num': FLAGS.user_hash_num,
    'item_hash_num': FLAGS.item_hash_num
}
model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_name)
user_dict = {}
user_dict["user_no"] = test_df.user_no.unique()
user_dict["item_no"] = np.zeros(len(user_dict["user_no"]))
test_df_tmp = pd.DataFrame(user_dict)
predictions = list(itertools.islice(model.predict(input_fn=eval_input_fn), 0, len(user_dict["user_no"]) + 2))

user_vec_map = {}
vec = set()
for i in range(len(predictions)):
    user_vec_map[predictions[i]["id"]] = predictions[i]["predict_result"]
id_index = []
id_vectors = []
for key in user_vec_map.keys():
    id_index.append(key)
    vector = np.array(user_vec_map[key].decode('utf8').split(',')).astype(np.float32)
    id_vectors.append(vector)
id_vectors = np.array(id_vectors)
id_index = np.array(id_index)
np.save("userid_vectors_m4.npy", id_vectors)
np.save("userid_index_m4.npy", id_index)
print("output user vector")
