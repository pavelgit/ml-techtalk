from flask import Flask, request, jsonify
from sources.DataProvider import DataProvider
from sources.NoOccupationNetModelProvider import NoOccupationNetModelProvider
from sources.NoOccupationPresenceModelProvider import NoOccupationPresenceModelProvider
from joblib import load
import numpy as np
import logging, sys
import datetime
import tensorflow as tf
from flask_cors import CORS

data_provider = DataProvider()

presence_model_provider = NoOccupationPresenceModelProvider()
presence_model_provider.load()

net_price_model_provider = NoOccupationNetModelProvider()
net_price_model_provider.load()

graph = tf.get_default_graph()

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
app = Flask(__name__)
CORS(app)

@app.route('/tariff-prices', methods=['POST'])
def calculate_prices():
    request_data = dict(request.get_json())

    request_data['benefitAgeLimit'] = int(request_data['benefitAgeLimit'])
    request_data['benefitAmount'] = int(request_data['benefitAmount'])
    request_data['fractionOfficeWork'] = int(request_data['fractionOfficeWork'])
    request_data['staffResponsibility'] = int(request_data['staffResponsibility'])

    request_data['birthday'] = datetime.datetime.strptime(request_data['birthday'], '%Y-%m-%d')
    request_data['insuranceStart'] = datetime.datetime.strptime(request_data['insuranceStart'], '%Y-%m-%d')
    input_data = list(data_provider.get_example_input_array_without_occupation(request_data))

    global graph
    is_present = None
    with graph.as_default():
        is_present = float(presence_model_provider.model.predict(np.array([input_data], dtype=float))[0][0])
        net_price = float(net_price_model_provider.model.predict(np.array([input_data], dtype=float))[0][0])

    response = {
        'is_present': is_present,
        'net_price': net_price
    }

    return jsonify(response), 200

app.run(debug=True)