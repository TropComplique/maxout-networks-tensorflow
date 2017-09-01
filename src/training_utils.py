import tensorflow as tf
import shutil
import os
import time


def train(run, graph, ops, X_train, Y_train, X_val, Y_val, batch_size,
          num_epochs, steps_per_epoch, validation_steps, patience=5,
          warm=False, initial_epoch=1, verbose=True):
    """Fit a defined network.

    Arguments:
        run: An integer that determines a folder where logs and the fitted model
            will be saved.
        graph: A Tensorflow graph.
        ops: A list of ops from the graph.
        X_train: A numpy array of shape [n_train_samples, n_features]
            and of type 'float32'.
        Y_train: A numpy array of shape [n_train_samples, n_classes]
            and of type 'float32'.
        X_val: A numpy array of shape [n_val_samples, n_features]
            and of type 'float32'.
        Y_val: A numpy array of shape [n_val_samples, n_classes]
            and of type 'float32'.
        batch_size: An integer.
        num_epochs: An integer.
        steps_per_epoch: An integer, number of optimization steps per epoch.
        validation_steps: An integer, number of batches from validation dataset
            to evaluate on.
        patience: An integer, number of epochs before early stopping if
            test logloss isn't improving.
        warm: Boolean, if `True` then resume training from the previously
            saved model.
        initial_epoch: An integer, epoch at which to start training
            (useful for resuming a previous training run).
        verbose: Boolean, whether to print train and test logloss/accuracy
            during fitting.

    Returns:
        losses: A list of tuples containing train and test logloss/accuracy.
        is_early_stopped: Boolean, if `True` then fitting is stopped early.

    """

    # create folders for logging and saving
    dir_to_log = 'logs/run' + str(run)
    dir_to_save = 'saved/run' + str(run)
    if os.path.exists(dir_to_log) and not warm:
        shutil.rmtree(dir_to_log)
    if os.path.exists(dir_to_save) and not warm:
        shutil.rmtree(dir_to_save)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter(dir_to_log, sess.graph)

    # get graph's ops
    data_init_op, predictions_op, log_loss_op, optimize_op,\
        grad_summaries_op, init_op, saver_op, accuracy_op, summaries_op = ops

    if warm:
        saver_op.restore(sess, dir_to_save + '/model')
    else:
        sess.run(init_op)

    # things that will be returned
    losses = []
    is_early_stopped = False

    training_epochs = range(
        initial_epoch,
        initial_epoch + num_epochs
    )

    # initialize data sources
    data_dict = {
        'input_pipeline/X_train:0': X_train,
        'input_pipeline/Y_train:0': Y_train,
        'input_pipeline/X_val:0': X_val,
        'input_pipeline/Y_val:0': Y_val,
        'input_pipeline/batch_size:0': batch_size
    }
    sess.run(data_init_op, data_dict)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # begin training
    for epoch in training_epochs:

        start_time = time.time()
        running_loss, running_accuracy = 0.0, 0.0

        # at zeroth step also collect metadata and summaries
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )
        run_metadata = tf.RunMetadata()
        # do epoch's zeroth step
        _, batch_loss, batch_accuracy, summary, grad_summary = sess.run(
            [optimize_op, log_loss_op, accuracy_op, summaries_op, grad_summaries_op],
            options=run_options, run_metadata=run_metadata
        )
        writer.add_run_metadata(run_metadata, str(epoch))
        writer.add_summary(summary, epoch)
        writer.add_summary(grad_summary, epoch)
        running_loss += batch_loss
        running_accuracy += batch_accuracy

        # main training loop
        for step in range(1, steps_per_epoch):

            _, batch_loss, batch_accuracy = sess.run(
                [optimize_op, log_loss_op, accuracy_op]
            )

            running_loss += batch_loss
            running_accuracy += batch_accuracy

        # evaluate on the validation set
        test_loss, test_accuracy = _evaluate(
            sess, validation_steps, log_loss_op, accuracy_op
        )
        train_loss = running_loss/steps_per_epoch
        train_accuracy = running_accuracy/steps_per_epoch

        if verbose:
            print('  {0}      {1:.3f} {2:.3f}   {3:.3f} {4:.3f}   {5:.3f}'.format(
                epoch, train_loss, test_loss,
                train_accuracy, test_accuracy, time.time() - start_time
            ))

        # collect all losses and accuracies
        losses += [(epoch, train_loss, test_loss, train_accuracy, test_accuracy)]

        # consider a possibility of early stopping
        if _is_early_stopping(losses, patience, 2):
            is_early_stopped = True
            break

    coord.request_stop()
    coord.join(threads)

    saver_op.save(sess, dir_to_save + '/model')
    sess.close()

    return losses, is_early_stopped


def _evaluate(sess, validation_steps, log_loss_op, accuracy_op):

    test_loss, test_accuracy = 0.0, 0.0
    for i in range(validation_steps):
        batch_loss, batch_accuracy = sess.run(
            [log_loss_op, accuracy_op], {'control/is_training:0': False}
        )
        test_loss += batch_loss
        test_accuracy += batch_accuracy

    test_loss /= validation_steps
    test_accuracy /= validation_steps

    return test_loss, test_accuracy


def predict_proba(graph, ops, run, X):
    """Predict probabilities with a fitted model.

    Arguments:
        graph: A Tensorflow graph.
        ops: A list of ops from the graph.
        run: An integer that determines a folder where a fitted model
            is saved.
        X: A numpy array of shape [n_samples, n_features]
            and of type 'float32'.

    Returns:
        predictions: A numpy array of shape [n_samples, n_classes]
            and of type 'float32'.

    """
    sess = tf.Session(graph=graph)

    # get graph's ops
    data_init_op, predictions_op, log_loss_op, optimize_op,\
        grad_summaries_op, init_op, saver_op, accuracy_op, summaries_op = ops
    # only predictions_op and saver_op are used here

    saver_op.restore(sess, 'saved/run' + str(run) + '/model')
    feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
    predictions = sess.run(predictions_op, feed_dict)
    sess.close()

    return predictions


# it decides if training must stop
def _is_early_stopping(losses, patience, index_to_watch):
    test_losses = [x[index_to_watch] for x in losses]
    if len(losses) > (patience + 4):
        # running average
        average = (test_losses[-(patience + 4)] +
                   test_losses[-(patience + 3)] +
                   test_losses[-(patience + 2)] +
                   test_losses[-(patience + 1)] +
                   test_losses[-patience])/5.0
        return test_losses[-1] > average
    else:
        return False
