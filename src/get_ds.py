#-------------------------------------------------------------------------------------------------------------------------------

#  PNEUMONIA DETECTION

# DETECTS IF A PERSON HAS PNEUMONIA OR NOT USING THE X-RAY IMAGES OF THEIR CHEST. USES CONVOLUTIONAL NEURAL NETWORK

# FILE NAME: GET_DS.PY

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Binary Classification

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED LIBRARIES
import requests

def getFromGoogleDrive(id, destination):
    URL = "https://drive.google.com/u/0/uc?id=1ui5gZWlEVry4e8TTLeywkaTuv6EcuKto&export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    file_id = '1ui5gZWlEVry4e8TTLeywkaTuv6EcuKto'
    destination = 'dataset/data.zip'
    getFromGoogleDrive(file_id, destination)





'''

subprocess.Popen('cd ..', shell = True)
subprocess.Popen('mkdir dataset', shell = True)

DATASET_URL = 'https://doc-04-cc-docs.googleusercontent.com/docs/securesc/sg60p0fslhlu02kmk7le0ljia5vq30r4/l0stftu3r4ul71t54o0r35qq1ut2p42i/1611509325000/17518686928399744585/17518686928399744585/1ui5gZWlEVry4e8TTLeywkaTuv6EcuKto?e=download&authuser=0&nonce=a39k578m46dga&user=17518686928399744585&hash=72ptjt0qjakaqtrppie64mo18hv5qpmu'
LOCAL_PATH = 'dataset.zip'

subprocess.Popen('curl https://doc-04-cc-docs.googleusercontent.com/docs/securesc/sg60p0fslhlu02kmk7le0ljia5vq30r4/l0stftu3r4ul71t54o0r35qq1ut2p42i/1611509325000/17518686928399744585/17518686928399744585/1ui5gZWlEVry4e8TTLeywkaTuv6EcuKto?e=download&authuser=0&nonce=a39k578m46dga&user=17518686928399744585&hash=72ptjt0qjakaqtrppie64mo18hv5qpmu',shell = True)

'''