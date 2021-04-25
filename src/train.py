# PNEUMONIA DETECTION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Binary Classification, Deep Learning, TensorFlow, Convolutional Neural Networks

# IMPORTING REQUIRED LIBRARIES
import os
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

DIR_TRAIN = './dataset/pneumonia_extended/train'
DIR_TEST = './dataset/pneumonia_extended/test'

DIR_TRAIN_NORMAL = './dataset/pneumonia_extended/train/NORMAL'
DIR_TRAIN_PNEUMONIA = './dataset/pneumonia_extended/train/PNEUMONIA'
DIR_TEST_NORMAL = './dataset/pneumonia_extended/test/NORMAL'
DIR_TEST_PNEUMONIA = './dataset/pneumonia_extended/test/PNEUMONIA'

ACC_THRESHOLD = 0.99
MODEL_PATH = './model/pneumonia-model'

# DATA PREPROCESSING
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
train_data = datagen_train.flow_from_directory(DIR_TRAIN,
                                               target_size = (150,150), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
test_data = datagen_test.flow_from_directory(DIR_TEST,
                                               target_size = (150,150), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)


# COUNTING NUMBER OF IMAGE SAMPLES 
print('Total normal images on Train set :', len(os.listdir(DIR_TRAIN_NORMAL)))
print('Total pneumonia images on Train set :', len(os.listdir(DIR_TRAIN_PNEUMONIA)))
print('Total normal images on Test set :', len(os.listdir(DIR_TEST_NORMAL)))
print('Total pneumonia images on Test set :', len(os.listdir(DIR_TEST_PNEUMONIA)))


# DEFINING NEURAL NETWORK FUNCTION
def pneumoniaModel():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (150,150,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size =(2,2)))

    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Dense(64,activation = 'relu'))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
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

model.summary()

# FITTING AND TRAINING THE MODEL
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_data, validation_data = test_data, epochs = 30,batch_size = 5)

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()
plt.savefig('graphs/loss_graph.png')

# PLOTTING THE GRAPH FOR TRAIN-ACCURACY AND VALIDATION-ACCURACY
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
plt.savefig('graphs/acc_graph.png')

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)


