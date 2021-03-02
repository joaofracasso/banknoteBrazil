import io

import torchvision.transforms as transforms
from PIL import Image

import onnxruntime as ort
import numpy as np

class_map = {
    0: "10 Reais Frente",
    1: "10 Reais Verso",
    2: "20 Reais Frente",
    3: "20 Reais Verso",
    4: "2 Reais Frente",
    5: "2 Reais Verso",
    6: "50 Reais Frente",
    7: "50 Reais Verso",
    8: "5 Reais Frente",
    9: "5 Reais Verso"
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

    filename = ["data/validation/10reaisFrente/compressed_0_2854543.jpeg",
                "data/validation/10reaisVerso/compressed_0_2175135.jpeg",
                "data/validation/2reaisFrente/compressed_0_1835891.jpeg",
                'data/validation/2reaisVerso/compressed_0_3752849.jpeg',
                "data/validation/5reaisFrente/compressed_0_1986857.jpeg",
                "data/validation/5reaisVerso/compressed_0_4651610.jpeg",
                'data/validation/20reaisFrente/compressed_0_1516768.jpeg',
                'data/validation/20reaisVerso/compressed_0_3080811.jpeg',
                'data/validation/50reaisFrente/compressed_0_1478513.jpeg',
                'data/validation/50reaisVerso/compressed_0_3923784.jpeg']
    for img in filename:
        with open(img, 'rb') as f:
            image_bytes = f.read()
            tensor = get_prediction(image_bytes, ort_session)
            print(tensor)
