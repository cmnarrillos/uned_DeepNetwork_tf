import pickle
import gzip
import numpy as np
import tensorflow as tf

def load_data_shared(filename="./data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    def shared(data):
        """Place the data into shared variables. This allows TensorFlow to access the data efficiently."""
        shared_x = tf.Variable(
            np.asarray(data[0], dtype=np.float64), trainable=False)
        shared_y = tf.Variable(
            np.asarray(data[1], dtype=np.int32), trainable=False)
        return shared_x, tf.cast(shared_y, tf.int32)
    return [shared(training_data), shared(validation_data), shared(test_data)]

# User-defined activation function
def ReLU_mod(x):
    return tf.maximum(x/1000, x)  # Example: Squaring the input
