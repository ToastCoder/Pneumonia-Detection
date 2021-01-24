# PNEUMONIA DETECTION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Binary Classification, Deep Learning, TensorFlow, Convolutional Neural Networks

# DISABLE TF DEBUGGING INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

# IMPORTING REQUIRED MODULES
import numpy as np
import tensorflow as tf

# GETTING THE TRAINED MODEL
model = tf.keras.models.load_model('./model/pneumonia-model')

# GETTING THE SAMPLE DATA FROM SAMPLES FOLDER
data_dir = './samples/'
images = []
for img in os.listdir(data_dir):
    img = os.path.join(data_dir, img)
    img = tf.keras.preprocessing.image.load_img(img, target_size=(200, 200), grayscale = True)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
files = os.listdir(data_dir)
images = np.vstack(images)

# PREDICTING THE CLASSIFICATION FOR EVERY FILE IN THE SAMPLES DIRECTORY
classes = model.predict_classes(images, batch_size=10)

# USER UNDERSTANDABLE OUTPUT FORMATTING
for i in range(len(classes)):
    if int(classes[i]) == 0:
        print(f"The image {files[i]} seems to be a report of a normal person. ")
    elif int(classes[i]) == 1:
        print(f"The image {files[i]} seems to be a report of a person with pneumonia. ")
