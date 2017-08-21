import tensorflow as tf
import math


def _nonlinearity(X):
    return tf.nn.relu(X, name='ReLU')


def _dropout(X, rate, is_training):
    keep_prob = tf.constant(
        1.0 - rate, tf.float32,
        [], 'keep_prob'
    )
    result = tf.cond(
        is_training,
        lambda: tf.nn.dropout(X, keep_prob),
        lambda: tf.identity(X),
        name='dropout'
    )
    return result


def _batch_norm(X, is_training):
    return tf.contrib.layers.batch_norm(
        X, is_training=is_training, center=True,
        scale=False, fused=True, scope='batch_norm'
    )


def _affine(X, size):
    input_dim = X.shape.as_list()[1]
    maxval = math.sqrt(2.0/input_dim)

    W = tf.get_variable(
        'weights', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    b = tf.get_variable(
        'bias', [size], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(tf.matmul(X, W), b)


def _maxout_layer(X, size, k):
    input_dim = X.shape.as_list()[1]
    maxval = math.sqrt(2.0/input_dim)

    W = tf.get_variable(
        'weights', [input_dim, k, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    b = tf.get_variable(
        'bias', [k, size], tf.float32,
        tf.zeros_initializer()
    )

    result = tf.add(tf.tensordot(X, W, [[1], [0]]), b)
    result = tf.reduce_max(result, axis=1, name='maxout')
    return result


def _mapping(X, architecture, dropout, is_training):
    # number of layers
    depth = len(architecture) - 1
    result = X

    for i in range(1, depth):
        with tf.variable_scope('layer_' + str(i)):
            size, k = architecture[i]
            result = _dropout(result, dropout[i - 1], is_training)
            result = _maxout_layer(result, size, k)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, result)
            result = _batch_norm(result, is_training)

    with tf.variable_scope('linear_layer'):
        result = _dropout(result, dropout[-1], is_training)
        logits = _affine(result, architecture[-1])

    return logits


def _add_weight_decay(weight_decay):

    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    weights = [v for v in trainable if 'weights' in v.name]

    for W in weights:
        l2_loss = tf.multiply(
            weight_decay, tf.nn.l2_loss(W), name='l2_loss'
        )
        tf.losses.add_loss(l2_loss)
