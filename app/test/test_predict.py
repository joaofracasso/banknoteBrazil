import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
import onnxruntime as ort

import pytest
from predict_model import get_prediction

files = [
    "data/test/2reaisFrente/compressed_0_1835891.jpeg",
    'data/test/2reaisVerso/compressed_20_9551306.jpeg',
    "data/test/5reaisFrente/compressed_0_1986857.jpeg",
    'data/test/5reaisVerso/compressed_0_4651610.jpeg',
    "data/test/10reaisFrente/compressed_0_2854543.jpeg",
    "data/test/10reaisVerso/compressed_0_2175135.jpeg",
    'data/test/20reaisFrente/compressed_0_1516768.jpeg',
    'data/test/20reaisVerso/compressed_0_3080811.jpeg',
    'data/test/50reaisFrente/compressed_0_1478513.jpeg',
    'data/test/50reaisVerso/compressed_0_3923784.jpeg']

@pytest.mark.parametrize('image', files)
def test_get_prediction(image):
    ort_session = ort.InferenceSession('app/models/banknote_best.onnx')
    with open(image, 'rb') as f:
        image_bytes = f.read()
        class_ = get_prediction(image_bytes, ort_session)
    assert isinstance(class_, str)
