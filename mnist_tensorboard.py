import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to match the input shape for the neural network
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Define a simple neural network model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 output classes for MNIST
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()

# Step 3: Enable TensorBoard logging
log_dir = "logs/fit/"  # Directory to save TensorBoard logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Step 4: Train the model with TensorBoard callback
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback], batch_size=64, verbose=2)

# Step 5: Launch TensorBoard and analyze loss and accuracy trends
# To visualize TensorBoard, run the following command in the terminal:
# tensorboard --logdir=logs/fit

# Displaying a quick comparison of training vs validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Displaying a quick comparison of training vs validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
