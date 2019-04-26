from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.INFO)
# logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

RNN_CELLSIZE = 80
N_LAYERS = 2
DROPOUT_PKEEP = 0.7
DATA_SEQ_LEN = 100000
SEQLEN = 20  # unrolled sequence length
BATCHSIZE = 32

X_train, X_test, Y_train, Y_test = None, None, None, None


def train_dataset():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(DATA_SEQ_LEN * 4 // SEQLEN)
    dataset = dataset.batch(BATCHSIZE)
    samples, labels = dataset.make_one_shot_iterator().get_next()
    return samples, labels


def eval_dataset():
    evaldataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    evaldataset = evaldataset.repeat(1)
    evaldataset = evaldataset.batch(BATCHSIZE)

    samples, labels = evaldataset.make_one_shot_iterator().get_next()
    return samples, labels


def model_rnn_fn(features, labels, mode):
    X = tf.expand_dims(features, axis=2)

    batchsize = tf.shape(X)[0]
    seqlen = tf.shape(X)[1]

    cells = [tf.nn.rnn_cell.GRUCell(RNN_CELLSIZE) for _ in range(N_LAYERS)]

    cells[:-1] = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=DROPOUT_PKEEP) for cell in cells[:-1]]
    # a stacked RNN cell still works like an RNN cell
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)

    # X[BATCHSIZE, SEQLEN, 1], Hin[BATCHSIZE, RNN_CELLSIZE*N_LAYERS]
    Yn, H = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    Yn = tf.reshape(Yn, [batchsize * seqlen, RNN_CELLSIZE])
    Yr = tf.layers.dense(Yn, 1)  # Yr [BATCHSIZE*SEQLEN, 1]
    Yr = tf.reshape(Yr, [batchsize, seqlen, 1])  # Yr [BATCHSIZE, SEQLEN, 1]

    Yout = Yr[:, -1, :]  # Last output Yout [BATCHSIZE, 1]

    loss = train_op = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.expand_dims(labels, axis=2)
        loss = tf.losses.mean_squared_error(Yr, labels)  # la  bels[BATCHSIZE, SEQLEN, 1]
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_op = tf.contrib.training.create_train_op(loss, optimizer)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"Yout": Yout},
        loss=loss,
        train_op=train_op
    )


def main():
    data = pd.read_csv("..//..//data//sens0-0.csv", header=None, delimiter=';')[1].values[:DATA_SEQ_LEN]
    data = np.array(data).astype(np.float32)
    global X_train, X_test, Y_test, Y_train

    X = data
    Y = np.roll(data, -1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=.2,
                                                        random_state=0)

    X_train = np.reshape(X_train, [-1, SEQLEN])
    Y_train = np.reshape(Y_train, [-1, SEQLEN])

    X_test = np.reshape(X_test, [-1, SEQLEN])
    Y_test = np.reshape(Y_test, [-1, SEQLEN])

    estimator = tf.estimator.Estimator(model_fn=model_rnn_fn)
    estimator.train(input_fn=train_dataset, steps=10000)

    results = estimator.predict(eval_dataset)

    pred = [result["Yout"] for result in results]

    actual = Y_test[:, -1]

    plt.plot(actual, label="Actual Values", color='green')
    plt.plot(pred, label="Predicted Values", color='red')
    plt.legend()
    print("MSE: {}".format(np.sqrt(mean_squared_error(pred, actual))))
    plt.show()


if __name__ == '__main__':
    main()









