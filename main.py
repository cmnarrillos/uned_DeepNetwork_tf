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
train_data_np = train_data.numpy().reshape(-1, 28, 28, 1)
train_data = tf.convert_to_tensor(train_data_np)
train_labels = train_labels.numpy()
train_labels = tf.convert_to_tensor(train_labels)

(val_data, val_labels) = validation_data
val_data_np = val_data.numpy().reshape(-1, 28, 28, 1)
val_data = tf.convert_to_tensor(val_data_np)
val_labels = val_labels.numpy()
val_labels = tf.convert_to_tensor(val_labels)

(test_data, test_labels) = test_data
test_data_np = test_data.numpy().reshape(-1, 28, 28, 1)
test_data = tf.convert_to_tensor(test_data_np)
test_labels = test_labels.numpy()
test_labels = tf.convert_to_tensor(test_labels)

# Initialize the neural network model
model = models.Sequential()

# Add convolutional layers
# model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.Conv2D(32, (5, 5), activation='relu'))
# model.add(layers.Conv2D(64, (5, 5), activation='relu'))

# Flatten the output for the fully connected layer
model.add(layers.Flatten())

# Add fully connected layer
model.add(layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(5.0/50000)))

# Add softmax layer
model.add(layers.Dense(10, activation='softmax'))

# Define optimizer and loss function
optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, batch_size=10, epochs=60, validation_data=(val_data, val_labels))

# Test the accuracy on test data
test_loss, test_accuracy = model.evaluate(test_data, test_labels)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
