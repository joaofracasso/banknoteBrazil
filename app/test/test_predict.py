import pytest
from app.src.modeling.predict_model import get_prediction

files = "data/test/2reaisVerso/compressed_20_9551306.jpeg"

@pytest.mark.parametrize('image', [files])
def test_get_prediction(image):
    with open(image, 'rb') as f:
        image_bytes = f.read()
        class_ = get_prediction(image_bytes)
    assert isinstance(class_, str)