import base64
import json

import onnxruntime as ort

from src.modeling.predict_model import get_prediction


def lambda_handler(event, context):
    ort_session = ort.InferenceSession('app/models/banknote_best.onnx')
    output = get_prediction(image_bytes=base64.b64decode(event["body"]), ort_session)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'Nota': output
        })
    }
