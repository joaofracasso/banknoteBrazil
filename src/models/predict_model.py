import io

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

import onnx

import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession('models/banknote.onnx')

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = ort_session.run(None, {'input.1': tensor.numpy()})
    y_hat = np.argmax(outputs, axis=1)
    return y_hat

if __name__ == "__main__":
    with open("/home/joaof/banknoteBrazil/src/models/10_front.jpg", 'rb') as f:
        image_bytes = f.read()
        tensor = get_prediction(image_bytes=image_bytes)
        print(tensor)