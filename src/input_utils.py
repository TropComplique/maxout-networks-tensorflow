import tensorflow as tf


# input pipeline
def _get_data(num_classes, input_dim, is_training):

    batch_size = tf.Variable(
        tf.placeholder(tf.int32, [], 'batch_size'),
        trainable=False, collections=[]
    )

    X_train, Y_train, train_x_batch, train_y_batch = _get_batch(
        num_classes, input_dim, batch_size
    )
    X_val, Y_val, val_x_batch, val_y_batch = _get_val_batch(
        num_classes, input_dim, batch_size
    )

    x_batch, y_batch = tf.cond(
        is_training,
        lambda: (train_x_batch, train_y_batch),
        lambda: (val_x_batch, val_y_batch)
    )

    init_data = tf.variables_initializer(
        [X_train, Y_train, X_val, Y_val, batch_size]
    )
    return init_data, x_batch, y_batch


# get batch for training
def _get_batch(num_classes, input_dim, batch_size):

    X_train = tf.Variable(
        tf.placeholder(tf.float32, [None, input_dim], 'X_train'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, input_dim]
    )
    Y_train = tf.Variable(
        tf.placeholder(tf.float32, [None, num_classes], 'Y_train'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, num_classes]
    )

    # three values that you need to tweak
    min_after_dequeue = 10000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 2)*128
    # 128 is a typical batch size

    x_batch, y_batch = tf.train.shuffle_batch(
        [X_train, Y_train], batch_size, capacity, min_after_dequeue,
        num_threads, enqueue_many=True,
        shapes=[[input_dim], [num_classes]]
    )

    return X_train, Y_train, x_batch, y_batch


# get batch for validation
def _get_val_batch(num_classes, input_dim, batch_size):

    X_val = tf.Variable(
        tf.placeholder(tf.float32, [None, input_dim], 'X_val'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, input_dim]
    )
    Y_val = tf.Variable(
        tf.placeholder(tf.float32, [None, num_classes], 'Y_val'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, num_classes]
    )

    num_threads = 1
    capacity = 10000

    # it cycles through the validation dataset
    # but i don't understand why it works
    x_batch, y_batch = tf.train.batch(
        [X_val, Y_val], batch_size,
        num_threads, capacity,
        enqueue_many=True,
        shapes=[[input_dim], [num_classes]]
    )
    # set validation_steps=len(X_val)/batch_size
    # so that at each evaluation we are using the whole validation dataset

    return X_val, Y_val, x_batch, y_batch
