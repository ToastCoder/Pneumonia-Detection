#-------------------------------------------------------------------------------------------------------------------------------

#  PNEUMONIA DETECTION

# DETECTS IF A PERSON HAS PNEUMONIA OR NOT USING THE X-RAY IMAGES OF THEIR CHEST. USES CONVOLUTIONAL NEURAL NETWORK

# FILE NAME: CHECK_DS.SH

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Binary Classification

#-------------------------------------------------------------------------------------------------------------------------------

if [ -d "dataset/pneumonia_extended" ]
then
    echo "Directory exists. Success!"
else
    echo "Directory isn't available. Checking for dataset.zip file..."
    if [ -e "dataset/dataset.zip" ]
    then
        unzip -q 'dataset/dataset.zip' -d './dataset'
        echo "Unzipped dataset successfully."
    else
        . src/get_ds.sh
        unzip -q 'dataset/dataset.zip' -d './dataset'
        echo "Unzipped dataset successfully."
    fi
fi
