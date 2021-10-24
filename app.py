import os
import io
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2
import pywt
from PIL import Image

from tensorflow import keras

from flask import Flask, jsonify, request 

shape = (512,512)
le = pickle.load(open(os.path.join("model","le.sav"),"rb"))
model = keras.models.load_model('model')
transdict = {
    "retak buaya": "Area Crack",
    "retak garis": "Line Crack",
    "tidak retak": "Good Condition"
}


def prepare_image(img):
    img = np.frombuffer(img, np.uint8)
    arr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    arr = cv2.resize(arr,shape)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(arr)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    arr = cv2.merge((cl,a,b))
    arr = cv2.cvtColor(arr, cv2.COLOR_LAB2BGR)
    arr = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(arr, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    return LL


def predict_result(img):
    X = np.array(img)/255
    input_shape = (X.shape[1],X.shape[2])
    X = X.reshape(-1, input_shape[0], input_shape[1], 1)
    predictions = model.predict(X)
    predictions = np.argmax(predictions, axis=-1)
    results = [transdict[p] for p in le.inverse_transform(predictions)]
    return results

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.getlist('file')
    if not file:
        return
    imgs = []
    for f in file:
        img_bytes = f.read()
        img = prepare_image(img_bytes)
        imgs.append(img)

    return jsonify(prediction=predict_result(imgs))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')