import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
import pytest
from src.modeling.predict_model import get_prediction

files = "data/test/2reaisVerso/compressed_20_9551306.jpeg"

@pytest.mark.parametrize('image', [files])
def test_get_prediction(image):
    with open(image, 'rb') as f:
        image_bytes = f.read()
        class_ = get_prediction(image_bytes)
    assert isinstance(class_, str)