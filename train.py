# PNEUMONIA DETECTION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Binary Classification, Deep Learning, TensorFlow, Convolutional Neural Networks

# DISABLE TF DEBUGGING INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

# IMPORTING REQUIRED LIBRARIES
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

DIR_TRAIN = './dataset/train'
DIR_TEST = './dataset/test'
ACC_THRESHOLD = 0.99
MODEL_PATH = './model/pneumonia-model'

# DATA PREPROCESSING
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
train_data = datagen_train.flow_from_directory(DIR_TRAIN,
                                               target_size = (200,200), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
test_data = datagen_test.flow_from_directory(DIR_TEST,
                                               target_size = (200,200), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)

# DEFINING NEURAL NETWORK FUNCTION
def pneumoniaModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = (200,200,1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model

# CALLBACK CLASS
class Callback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACC_THRESHOLD):   
            print("Reached Threshold Accuracy, Stopping Training.")   
            self.model.stop_training = True

model = pneumoniaModel()
callback = Callback()

# FITTING AND TRAINING THE MODEL
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_data, validation_data = test_data, epochs = 20,callbacks = [callback])

# VISUALIZING TRAINING LOSS AND VALIDATION LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)


