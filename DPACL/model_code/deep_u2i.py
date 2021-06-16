import tensorflow as tf

def model_fn(features, labels, mode, params):
    with tf.name_scope('user_embeddings'):
        user_no_st = tf.cast(features['user_no'], dtype=tf.int64)
        user_no_embeddings = tf.Variable(
            tf.random_uniform([params["user_hash_num"], params['embedding_size']], -0.1, 0.1))
        user_net = tf.nn.embedding_lookup(user_no_embeddings, user_no_st)
        user_norm = tf.sqrt(tf.reduce_sum(tf.square(user_net), axis=1, keep_dims=True))
        user_net = tf.truediv(user_net, user_norm + 1e-9)

    with tf.name_scope('item_embeddings'):
        item_no_st = tf.cast(features['item_no'], dtype=tf.int64)
        item_no_embeddings = tf.Variable(
            tf.random_uniform([params["item_hash_num"], params['embedding_size']], -0.1, 0.1))
        item_net = tf.nn.embedding_lookup(item_no_embeddings, item_no_st)
        item_norm = tf.sqrt(tf.reduce_sum(tf.square(item_net), axis=1, keep_dims=True))
        item_net = tf.truediv(item_net, item_norm + 1e-9)

    with tf.name_scope('cosine_similarity'):
        if mode == tf.estimator.ModeKeys.PREDICT and params['predict_type'] == "user":
            return tf.estimator.EstimatorSpec(mode=mode, predictions={
                "predict_result": tf.reduce_join(tf.as_string(user_net), axis=1, separator=','),
                'id': tf.reshape(features['user_no'], [-1])})
        if mode == tf.estimator.ModeKeys.PREDICT and params['predict_type'] == "item":
            return tf.estimator.EstimatorSpec(mode=mode, predictions={
                "predict_result": tf.reduce_join(tf.as_string(item_net), axis=1, separator=','),
                'id': tf.reshape(features['item_no'], [-1])})
        # softmax
        user_item_cos = tf.matmul(user_net, tf.transpose(item_net)) * 20

    with tf.name_scope('loss'):
        labels = tf.range(tf.shape(user_net)[0])
        predicted_classes = tf.argmax(user_item_cos, 1)
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='accuracy_op')
        metrics = {'accuracy': accuracy}
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=user_item_cos)
        # loss = match_loss + ae_loss + item_l2norm
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

    with tf.name_scope('train'):
        optimizer = params['optimizer'](params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

