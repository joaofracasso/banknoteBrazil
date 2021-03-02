import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
import onnxruntime as ort

import pytest
from src.modeling.predict_model import get_prediction

files = "data/test/2reaisVerso/compressed_20_9551306.jpeg"

@pytest.mark.parametrize('image', [files])
def test_get_prediction(image):
    ort_session = ort.InferenceSession('app/models/banknote_best.onnx')
    with open(image, 'rb') as f:
        image_bytes = f.read()
        class_ = get_prediction(image_bytes, ort_session)
    assert isinstance(class_, str)
