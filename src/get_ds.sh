#-------------------------------------------------------------------------------------------------------------------------------

#  PNEUMONIA DETECTION

# DETECTS IF A PERSON HAS PNEUMONIA OR NOT USING THE X-RAY IMAGES OF THEIR CHEST. USES CONVOLUTIONAL NEURAL NETWORK

# FILE NAME: GET_DS.SH

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Binary Classification

#-------------------------------------------------------------------------------------------------------------------------------


if [ ! -d "dataset" ]
then
    mkdir dataset
fi

if [ ! -e "dataset/dataset.zip" ]
then
    cd dataset
    wget --no-check-certificate 'https://storage.googleapis.com/kaggle-data-sets/1119456/1880008/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210126%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210126T063117Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=3aaabe8ef0dde63b3a3c47b9ba3dd2c3ac33a73e3af743fc907433dc7741df8d092d560155de3754ba62d624e11e0b4d9c14442892ad2994b76ae0d494fd5b20e3706e80d2f9be98d06a660c29c0c42f7ad5c0c79a7bc1df8f432a20619337d91749a0b472ff64efccfdc9bb4e10d7411535cc95376c47398965942077ad0e7d687530bee22cafef03696961f03a4faf049a4a9791b4d07e3c03e07aee0e07aa2b1bdbdb1ca0f2e619c563bd124eb46f2adcbc301dabeb49c5fd97e81ac0e53bbfbf7597b67d1faf39db0f7020df647fb4f207b5f4e180604c58c6588eaf9cb36d5f73e0d7570606b99ba4f9f4a6f864e4245164ee96d71222be0326b90e6c08' -O 'dataset.zip'
    cd ..
else
    echo "dataset.zip exists."
fi

