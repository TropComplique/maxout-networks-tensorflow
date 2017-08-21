import tensorflow as tf
from input_utils import _get_data
from parts_of_the_net import _mapping, _add_weight_decay


def get_maxout_network(architecture, dropout, optimizer, weight_decay=None):
    """Create a maxout network computational graph.

    Arguments:
        architecture: A list that contains number of hidden units for each layer,
            where architecture[0] equals to the number of input features,
            architecture[-1] equals to the number of classes,
            architecture[i] has form (n_hidden_units, k),
            k - number of affine feature maps.
        dropout: A list that contains dropout rate for each layer.
            It is required that len(dropout) == len(architecture) - 1.
        optimizer: A Tensorflow optimizer.
        weight_decay: A scalar or None.

    For example:
        architecture=[54, (80, 3), (80, 3), (80, 3), 7],
        dropout=[0.2, 0.5, 0.5, 0.1]

    See arxiv.org/abs/1302.4389 for a description of maxout networks.

    """

    graph = tf.Graph()
    with graph.as_default():

        with tf.variable_scope('control'):
            is_training = tf.placeholder_with_default(True, [], 'is_training')

        input_dim = architecture[0]
        num_classes = architecture[-1]

        with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
            data_init, x_batch, y_batch = _get_data(num_classes, input_dim, is_training)

        with tf.variable_scope('inputs'):
            X = tf.placeholder_with_default(x_batch, [None, input_dim], 'X')
            Y = tf.placeholder_with_default(y_batch, [None, num_classes], 'Y')

        logits = _mapping(X, architecture, dropout, is_training)

        with tf.variable_scope('softmax'):
            predictions = tf.nn.softmax(logits)

        with tf.variable_scope('log_loss'):
            log_loss = tf.losses.softmax_cross_entropy(Y, logits)

        if weight_decay is not None:
            with tf.variable_scope('weight_decay'):
                _add_weight_decay(weight_decay)

        with tf.variable_scope('total_loss'):
            total_loss = tf.losses.get_total_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
            grads_and_vars = optimizer.compute_gradients(total_loss)
            optimize = optimizer.apply_gradients(grads_and_vars)

        grad_summaries = tf.summary.merge(
            [tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
             for g, v in grads_and_vars]
        )

        with tf.variable_scope('utilities'):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            is_equal = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

        summaries = _add_summaries()

    graph.finalize()
    ops = [
        data_init, predictions, log_loss, optimize,
        grad_summaries, init, saver, accuracy, summaries
    ]
    return graph, ops


def _add_summaries():
    # add histograms of all trainable variables and of all layer activations

    summaries = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)

    for v in trainable_vars:
        summaries += [tf.summary.histogram(v.name[:-2] + '_hist', v)]
    for a in activations:
        summaries += [tf.summary.histogram(a.name[:-2] + '_activ_hist', a)]

    return tf.summary.merge(summaries)
