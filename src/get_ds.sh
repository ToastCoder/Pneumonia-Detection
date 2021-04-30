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
    wget --no-check-certificate 'https://storage.googleapis.com/kaggle-data-sets/1272/2280/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210407%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210407T105916Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=7aa859272a33c022e3f545e45b67480afa6e14cd5416264b0675e92228ba9562027f221ce6cc65aaa40c3a1424cc04950687a09beb48f50b3b69692b9ad981a1cc1f41ad08f9e2bc62e8a21ae4d49057dc7cc911dfc3e9790d70a7a7ad2250af4867ba0408ebf1a50b4f16e2d83af026a48411a7561280d2ea8961cb87c42967e2e8be420966e8c66fa3ba6f400220d3ba9eca4ad1e776553a5a85dac267974202d4ffaf29fa45b9a6f89b4e832eb265fb6ad824f0affcc2575e79c9e3d354fd5ad6ea1921982b24a203f680eb03604c2a97d9e564cbe1105d7f88b4541043e2ff30773ea814a215f9c43497df4adb69cf93269cbdbfd9ad5f4f64190cfd4a00' -O 'dataset.zip'
    cd ..
else
    echo "dataset.zip exists."
fi

