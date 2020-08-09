from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from models import Baseline
import torch.optim as optim
import torch.nn as nn

train_path = 'data/train'
valid_path = 'data/validation'


def load_dataset(data_path):
    transform = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
                                  )
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1024,
        num_workers=0,
        shuffle=True)
    return train_loader


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    epochs = 1
    print(device)
    net = Baseline()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        with tqdm(total=len(load_dataset(train_path))) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch}')
            running_loss = 0.0
            running_val_loss = 0.0
            for i, data in enumerate(load_dataset(train_path)):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].to(device)
                labels = data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                loss.backward()
                optimizer.step()
                # Updating progress bar
                desc = f'Epoch {epoch} - loss {running_loss:.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(inputs.shape[0])

            for i, data_val in enumerate(load_dataset(valid_path)):
                inputs_val = data_val[0].to(device)
                labels_val = data_val[1].to(device)

                outputs_val = net(inputs_val)
                val_loss = criterion(outputs_val, labels_val)
                running_val_loss += val_loss.item()
            desc = (f'Epoch {epoch} - ' +
                    f'loss {running_loss:.4f} - ' +
                    f'val_loss  {running_val_loss:.4f}')
            epoch_pbar.set_description(desc)


    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(net, dummy_input, "models/banknote.onnx", verbose=True)
    print('Finished Training')
