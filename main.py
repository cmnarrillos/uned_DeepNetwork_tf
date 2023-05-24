import datetime
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from functions_deepnetwork import *

# # Set CPU as the device for TensorFlow (uncomment if needed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = current_time.strftime("%Y-%m-%d_%H-%M-%S")+'_deep'

# Load MNIST data
training_data, validation_data, test_data = load_data_shared()

# Convert data to proper inputs
(train_data, train_labels) = training_data
train_data = tf.convert_to_tensor(train_data.numpy().reshape(-1, 28, 28, 1))
train_labels = tf.convert_to_tensor(tf.expand_dims(train_labels, axis=1))

(val_data, val_labels) = validation_data
val_data = tf.convert_to_tensor(val_data.numpy().reshape(-1, 28, 28, 1))
val_labels = tf.convert_to_tensor(tf.expand_dims(val_labels, axis=1))

(test_data, test_labels) = test_data
test_data = tf.convert_to_tensor(test_data.numpy().reshape(-1, 28, 28, 1))
test_labels = tf.convert_to_tensor(tf.expand_dims(test_labels, axis=1))





# -----------------------------------------------------------------------------
# # 1st network to train: 1 hidden layer with 100 neurons:
if False:
    # Initialize the neural network model
    print('\n\n\n\n NEW CASE: 1 FullyConnected Layer')
    print('Architecture: [784, 100, 10]')
    model = models.Sequential([
        layers.Flatten(),
        layers.Dense(100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(5.0/50000)),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, batch_size=10, epochs=60,
              validation_data=(val_data, val_labels))

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

# -----------------------------------------------------------------------------
# # 2nd network to train: 1 conv-pool + 1 FC layer
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 100, 10]')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(5.0/50000)),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, batch_size=10, epochs=60,
              validation_data=(val_data, val_labels))

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)


    # net = Network([
    #     ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
    #                   filter_shape=(20, 1, 5, 5),
    #                   poolsize=(2, 2)),
    #     FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
    #     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    #
    # # Train the network
    # net.SGD(training_data, epochs, mini_batch_size, lr,
    #             validation_data, test_data, lmbda=lmbda)
