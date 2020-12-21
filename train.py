# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# DATA PREPROCESSING

dir_train = './dataset/train'
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
train_data = datagen_train.flow_from_directory(dir_train,
                                               target_size = (200,200), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)
dir_test = './dataset/test'
datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
test_data = datagen_test.flow_from_directory(dir_test,
                                               target_size = (200,200), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)

# DEFINING NEURAL NETWORK
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = (200,200,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

# FITTING AND TRAINING THE MODEL
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_data, validation_data = test_data, epochs = 10)

# VISUALIZING TRAINING LOSS AND VALIDATION LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# SAVING THE MODEL
tf.keras.models.save_model('./model/pneumonia-model')


