import io

import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    epochs = 100
    print(device)
    net = Baseline()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
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

print('Finished Training')
