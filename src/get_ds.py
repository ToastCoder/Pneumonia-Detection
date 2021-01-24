#-------------------------------------------------------------------------------------------------------------------------------

#  PNEUMONIA DETECTION

# DETECTS IF A PERSON HAS PNEUMONIA OR NOT USING THE X-RAY IMAGES OF THEIR CHEST. USES CONVOLUTIONAL NEURAL NETWORK

# FILE NAME: GET_DS.PY

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Binary Classification

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED LIBRARIES
import subprocess

subprocess.Popen('cd ..', shell = True)
subprocess.Popen('mkdir dataset', shell = True)
subprocess.Popen('cd dataset', shell = True)

DATASET_URL = 'https://doc-04-cc-docs.googleusercontent.com/docs/securesc/sg60p0fslhlu02kmk7le0ljia5vq30r4/l0stftu3r4ul71t54o0r35qq1ut2p42i/1611509325000/17518686928399744585/17518686928399744585/1ui5gZWlEVry4e8TTLeywkaTuv6EcuKto?e=download&authuser=0&nonce=a39k578m46dga&user=17518686928399744585&hash=72ptjt0qjakaqtrppie64mo18hv5qpmu'
LOCAL_PATH = 'dataset/dataset.zip'

