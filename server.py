import tensorflow.contrib.keras as keras
import json
import io
import base64
from flask import Flask, request
from PIL import Image

app = Flask(__name__)


def get_model():
    global model
    model = keras.models.load_model('model_name.h5')
    print('* O modelo foi carregado')


print('* Carregando Model')
get_model()


@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json()
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    prediction = model.predict(image)
    response = {
        'prediction': {
            'cachorro': prediction[0][0],
            'gato': prediction[0][1]
        }
    }
    return json.dumps(response)
