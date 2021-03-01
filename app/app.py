import json
import base64

from src.modeling.predict_model import get_prediction


def lambda_handler(event, context):
    output = get_prediction(image_bytes=base64.b64decode(event["body"]))
    return {
        'statusCode': 200,
        'body': json.dumps({
            'Nota': output
        })
    }
