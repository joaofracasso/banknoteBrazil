![Python application](https://github.com/joaofracasso/banknoteBrazil/workflows/Python%20application/badge.svg)
# banknoteBrazil: Brazilian paper money
This repository contains a classification of Brazilian paper money.

## Training image classification Brazilian paper money model

You can download the freely available datasets with the provided script (it may take a while):

```bash
$ sh app/src/data/download_validation.sh  
```

### Requirements
Python 3.8 or later with all [requirements.txt](https://github.com/joaofracasso/banknoteBrazil/blob/master/app/requirements.txt) dependencies installed. To install run:

```bash
$ python -m pip install -r app/requirements.txt
```

### Environments

BanknoteBrazil may be run in any of the following up-to-date verified environments ([Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Codespace** See [Codespace](https://github.com/features/codespaces)
- **VS Code** See [Vs Code](https://code.visualstudio.com/docs/remote/containers)

### Train the network

You can train the network with the `train.py` script. For more usage information see [this](train.py). To train with the default parameters:

```bash
$ python app/src/modeling/train_model.py
```

### Evaluating the model

Also, you can evaluate the model against the validation set

```bash
$ python app/src/modeling/evaluate_model.py
```

## Predicting the outputs

To predict the outputs of a trained model using some dataset:

```bash
$  python app/src/modeling/predict_model.py --file data/test/2reaisVerso/compressed_20_9551306.jpeg
```

## Deploy on lambda container

Build the app Dockerfile:

```bash
docker build --pull --rm -f "app/Dockerfile" -t banknotebrazil:latest "app" 
```

Run the app of bankNote:

```bash
docker run -p 8080:8080 banknotebrazil:latest
```

Send send an (base64) image over a POST request:

```bash
curl --location --request POST 'http://localhost:8080/2015-03-31/functions/function/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "body": "image/jpeg;base64"  
}' 
```

## Maintenance

This project is currently maintained by João Victor Calvo Fracasso and is for academic research use only. If you have any questions, feel free to contact joao.fracasso@outlook.com.

## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.