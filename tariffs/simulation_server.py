from flask import Flask
from flask_restful import Api, Resource, reqparse
from sources.DataProvider import DataProvider
from sources.NoOccupationNetModelProvider import NoOccupationNetModelProvider
from sources.NoOccupationPresenceModelProvider import NoOccupationPresenceModelProvider
from joblib import load
import numpy as np

app = Flask(__name__)
api = Api(app)
data_provider = DataProvider()
presence_model_provider = NoOccupationPresenceModelProvider()
presence_model_provider.load()

net_price_model_provider = NoOccupationNetModelProvider()
net_price_model_provider.load()

autosklearn_model = load('autosklearn_model.joblib')

class TariffPrice(Resource):
    def post(self, name):

        parser = reqparse.RequestParser()
        parser.add_argument("familyStatus")
        parser.add_argument("educationType")
        parser.add_argument("jobSituation")
        parser.add_argument("industry")
        parser.add_argument("benefitAgeLimit")
        parser.add_argument("benefitAmount")
        parser.add_argument("fractionOfficeWork")
        parser.add_argument("staffResponsibility")
        parser.add_argument("birthday")
        parser.add_argument("insuranceStart")
        args = parser.parse_args()

        input_data = data_provider.get_example_input_array_without_occupation(np.array(args))

        is_present = presence_model_provider.model.predict(input_data)[0]
        net_price = net_price_model_provider.model.predict(input_data)[0]
        autosklearn_net_price = autosklearn_model.predict(np.array([args]))[0][0]

        response = {
            is_present: is_present,
            net_price: net_price,
            autosklearn_net_price: autosklearn_net_price
        }
        return response, 200
      
api.add_resource(TariffPrice, "/tariff-prices/")

app.run(debug=True)