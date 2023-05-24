import datetime
import os
import shutil
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from functions_deepnetwork import load_data_shared
from functions_deepnetwork import ReLU_mod

# # Set CPU as the device for TensorFlow (uncomment if needed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = current_time.strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists('./tests/' + id_test):
    os.makedirs('./tests/' + id_test)
f = open('./tests/' + id_test + '/register.txt', 'w', newline='\r\n')

# Set default parameters for training all examples
epochs = 20
mini_batch_size = 100
lr = 0.1
lmbda = 5.0
dropout = 0.2

f.write('\nGENERAL SETTINGS')
f.write('\nepochs: ' + str(epochs))
f.write('\nmini_batch_size: ' + str(mini_batch_size))
f.write('\nlr: ' + str(lr))
f.write('\nlmbda: ' + str(lmbda))
f.write('\ndropout: ' + str(dropout))

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

expanded_training_data, _, _ = load_data_shared(
    './data/mnist_expanded.pkl.gz')
(expanded_train_data, expanded_train_labels) = expanded_training_data
expanded_train_data = tf.convert_to_tensor(
    expanded_train_data.numpy().reshape(-1, 28, 28, 1))
expanded_train_labels = tf.convert_to_tensor(
    tf.expand_dims(expanded_train_labels, axis=1))


# -----------------------------------------------------------------------------
# # 1st network to train: 1 hidden layer with 100 neurons:
if False:
    n = train_labels.shape[0]
    # Initialize the neural network model
    print('\n\n\n\n NEW CASE: 1 FullyConnected Layer')
    print('Architecture: [784, 100, 10]')
    f.write('\n\n\n\n\n NEW CASE: 1 FullyConnected Layer')
    f.write('\nArchitecture: [784, 100, 10]')
    model = models.Sequential([
        layers.Flatten(),
        layers.Dense(100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(train_data, train_labels, batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 2nd network to train: 1 conv-pool + 1 FC layer
if False:
    n = train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 100, 10]')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + FC Layer')
    f.write('\nArchitecture: [784, 20x(24,24), 100, 10]')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(train_data, train_labels, batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 3rd network to train: 2 conv-pool + 1 FC layer
if False:
    n = train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + FC Layer')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(train_data, train_labels, batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 4th network to train: 2 conv-pool + 1 FC layer with ReLU
if False:
    n = train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + FC Layer (ReLU)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + FC Layer (ReLU)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation='relu',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(train_data, train_labels, batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 5th network to train: 2 conv-pool + 1 FC layer with modified ReLU
if False:
    n = train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + FC Layer (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + FC Layer (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(train_data, train_labels, batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 6th network to train: 2 conv-pool + 1 FC layer with modified ReLU
# # Expanding training data to 250.000
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    print('Expanded training data')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + FC Layer')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    f.write('\nExpanded training data')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 7.1th network to train: 2 conv-pool + 2 FC layers with sigmoid
# # Expanding training data to 250.000
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 2 FC Layers (sigmoid)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                         '100, 100, 10]')
    print('Expanded training data')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 2 FC Layers (sigmoid)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                             '100, 100, 10]')
    f.write('\nExpanded training data')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 7.2th network to train: 2 conv-pool + 2 FC layers with ReLU
# # Expanding training data to 250.000
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 2 FC Layers (ReLU)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                         '100, 100, 10]')
    print('Expanded training data')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 2 FC Layers (ReLU)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                             '100, 100, 10]')
    f.write('\nExpanded training data')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation='relu',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(units=100, activation='relu',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 7.3th network to train: 2 conv-pool + 2 FC layers with modified ReLU
# # Expanding training data to 250.000
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 2 FC Layers (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                          '100, 100, 10]')
    print('Expanded training data')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 2 FC Layers (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                            '100, 100, 10]')
    f.write('\nExpanded training data')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 8.1th network to train: 2 conv-pool + 2 FC layers with sigmoid
# # Expanding training data to 250.000. Include dropout
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 2 FC Layers (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                          '100, 100, 10]')
    print('Expanded training data')
    print('Dropout')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 2 FC Layers (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                            '100, 100, 10]')
    f.write('\nExpanded training data')
    f.write('\nDropout')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation='sigmoid',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 8.2th network to train: 2 conv-pool + 2 FC layers with ReLU
# # Expanding training data to 250.000. Include dropout
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 2 FC Layers (ReLU)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                         '100, 100, 10]')
    print('Expanded training data')
    print('Dropout')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 2 FC Layers (ReLU)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                            '100, 100, 10]')
    f.write('\nExpanded training data')
    f.write('\nDropout')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation='relu',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation='relu',
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 8.3th network to train: 2 conv-pool + 2 FC layers with modified ReLU
# # Expanding training data to 250.000. Include dropout
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 2 FC Layers (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                          '100, 100, 10]')
    print('Expanded training data')
    print('Dropout')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 2 FC Layers (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                            '100, 100, 10]')
    f.write('\nExpanded training data')
    f.write('\nDropout')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 9th network to train: 2 conv-pool + 3 FC layers with modified ReLU
# # Expanding training data to 250.000. Include dropout
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 3 FC Layers (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                     '100, 100, 100, 10]')
    print('Expanded training data')
    print('Dropout')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 3 FC Layers (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                       '100, 100, 100, 10]')
    f.write('\nExpanded training data')
    f.write('\nDropout')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 10h network to train: 2 conv-pool + 4 FC layers with modified ReLU
# # Expanding training data to 250.000. Include dropout
if False:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 4 FC Layers (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                                '100, 100, 100, 100, 10]')
    print('Expanded training data')
    print('Dropout')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 4 FC Layers (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                                  '100, 100, 100, 100, 10]')
    f.write('\nExpanded training data')
    f.write('\nDropout')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


# -----------------------------------------------------------------------------
# # 11h network to train: 2 conv-pool + 5 FC layers with modified ReLU
# # Expanding training data to 250.000. Include dropout
if True:
    n = expanded_train_labels.shape[0]
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
          'Convolutional + Pool + 5 FC Layers (ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                           '100, 100, 100, 100, 100, 10]')
    print('Expanded training data')
    print('Dropout')
    f.write('\n\n\n\n\n NEW CASE: Convolutional + Pool + '
            'Convolutional + Pool + 5 FC Layers (ReLU_mod)')
    f.write('\nArchitecture: [784, 20x(24,24), 20x(12,12), '
                              '100, 100, 100, 100, 100, 10]')
    f.write('\nExpanded training data')
    f.write('\nDropout')
    model = models.Sequential([
        layers.Conv2D(filters=20, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=40, kernel_size=(5, 5),
                      activation='relu',
                      kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dropout(rate=dropout),
        layers.Dense(units=100, activation=ReLU_mod,
                     kernel_regularizer=regularizers.l2(lmbda/(2*n))),
        layers.Dense(10, activation='softmax')
    ])

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Train the model
    t_i = time.time()
    model.fit(expanded_train_data, expanded_train_labels,
              batch_size=mini_batch_size,
              epochs=epochs, validation_data=(val_data, val_labels))
    elapsed = time.time() - t_i

    # Test the accuracy on test data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    print('Elapsed time: ' + str(elapsed) + ' s')
    f.write('\nTest Loss: ' + str(test_loss))
    f.write('\nTest Accuracy: ' + str(test_accuracy))
    f.write('\nElapsed time: ' + str(elapsed) + ' s')


f.close()
try:
    shutil.copy('./console/log.txt', './tests/' + id_test + '/console_log.txt')
except:
    pass
