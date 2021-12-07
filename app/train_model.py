from __future__ import division, print_function

import copy
import os
import time
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import initialize_model

#   to the ImageFolder structure
data_dir = "data"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"
# Number of classes in the dataset
num_classes = 10
# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Number of epochs to train for 
num_epochs = 20
# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True
learning_rate = 0.001


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss, train_acc = 0.0 , 0.0
    for batch_idx, _data in enumerate(train_loader):
        inputs, labels = _data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss.backward()

        optimizer.step()


        train_loss += loss.item()
        train_acc += torch.sum(preds == labels.data)
    
    print('Epoch {}\tTrain set: Loss: {:.4f}\t Acc: {:.4f}'.format(epoch, train_loss/len(train_loader.dataset), train_acc/len(train_loader.dataset)))
    return train_loss/len(train_loader.dataset), train_acc/len(train_loader.dataset)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            inputs, labels = _data
            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs) # (batch, time, n_class)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            test_loss += loss.item()
            test_acc += torch.sum(preds == labels.data)
            
    print('\tTest set: Loss: {:.4f}\t Acc: {:.4f}'.format(test_loss/len(test_loader.dataset), test_acc/len(test_loader.dataset)))
    return test_loss

if __name__ == "__main__":
    # Top level data directory. Here we assume the format of the directory conforms 
    hparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs
    }
    best_loss = 9999
    wandb.init(project="banknote", entity="joaofracasso")
    torch.manual_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft, input_size = initialize_model(model_name, num_classes,
                                            feature_extract, 
                                            use_pretrained=True)  
                                            
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
    train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader =  DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=True, num_workers=4)
    wandb.config = hparams
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()   
    wandb.watch(model_ft) 
    #model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name == "inception"))
    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        train_loss, train_acc = train(model_ft, device, train_loader, criterion, optimizer_ft, epoch)
        test_loss = test(model_ft, device, test_loader, criterion)
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss
             })
        if test_loss < best_loss:
            x = torch.randn(1, 3, input_size, input_size).to(device)
            torch.onnx.export(model_ft,
                                x,
                                "app/models/banknote_best.onnx")
            wandb.save("app/models/banknote_best.onnx")
            best_loss = test_loss