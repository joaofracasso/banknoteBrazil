import io
import argparse

import torchvision.transforms as transforms
from PIL import Image

import onnxruntime as ort
import numpy as np


class_map = {
    0: "10Reais Frente",
    1: "10Reais Verso",
    2: "20Reais Frente",
    3: "20Reais Verso",
    4: "2Reais Frente",
    5: "2Reais Verso",
    6: "50Reais Frente",
    7: "50Reais Verso",
    8: "5Reais Frente",
    9: "5Reais Verso"
            }
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze_(0)


def get_prediction(image_bytes, inference_session):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = inference_session.run(None, {'input.1': tensor.numpy()})
    y_hat = np.argmax(outputs[0], axis=1)[0]
    return class_map[y_hat]


if __name__ == "__main__":
    
    ort_session = ort.InferenceSession('app/models/banknote_best.onnx')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='app/models/banknote_best.onnx', type=str)
    parser.add_argument('--file', default=None, type=str)
    opt = parser.parse_args()
    with open(opt.file, 'rb') as f:
        image_bytes = f.read()
        tensor = get_prediction(image_bytes, ort_session)
        print(tensor)
