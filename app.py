import json

from requests import get

from flask import Flask, jsonify, request, redirect

from src.models.predict_model import get_prediction

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id})

@app.route('/', methods=['GET'])
def health_check():
    if request.method == 'GET':
        return jsonify({'Status': "OK"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)