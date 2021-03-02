import os

import onnxruntime as ort
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data_dir = "data"
input_size = [224, 224]
batch_size = 1

if __name__ == "__main__":
    ort_session = ort.InferenceSession('app/models/banknote_best.onnx')
    data_transforms = {
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['validation']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['validation']}
    outputs = []
    labels = []
    for inputs, label in dataloaders_dict["validation"]:
        output = ort_session.run(None, {'input.1': inputs.numpy()})
        output = np.argmax(output[0], axis=1)[0]
        outputs.append(output)
        labels.append(label.data.tolist())
    labels = sum(labels, [])
    print(classification_report(labels, outputs))
