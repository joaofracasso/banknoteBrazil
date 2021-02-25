import io

import torchvision.transforms as transforms
from PIL import Image

import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession('app/models/banknote_best.onnx')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                        transforms.CenterCrop(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = ort_session.run(None, {'input.1': tensor.numpy()})
    y_hat = np.argmax(outputs[0], axis=1)
    return y_hat.item()


if __name__ == "__main__":
    filename = ["data/validation/2reaisVerso/compressed_20_9551306.jpeg",
                'data/validation/2reaisVerso/compressed_0_3752849.jpeg',
                "data/validation/5reaisFrente/compressed_0_1986857.jpeg"]
    for img in filename:
        with open(img, 'rb') as f:
            image_bytes = f.read()
            tensor = get_prediction(image_bytes=image_bytes)
            print(tensor)
