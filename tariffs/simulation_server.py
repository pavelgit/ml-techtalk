from flask import Flask
from flask_restful import Api, Resource, reqparse
from sources.DataProvider import DataProvider
from sources.NoOccupationNetModelProvider import NoOccupationNetModelProvider
from joblib import load
import numpy as np

app = Flask(__name__)
api = Api(app)
data_provider = DataProvider()
model_provider = NoOccupationNetModelProvider()
model_provider.load()
autosklearn_model = load('model.joblib')

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
        output_data = model_provider.model.predict(input_data)
        my_net_price = output_data[0]

        autosklearn_output_data = autosklearn_model.predict(np.array([args]))
        autosklearn_net_price = autosklearn_output_data[0][0]

        response = {
            my_net_price: my_net_price,
            autosklearn_net_price: autosklearn_net_price
        }
        return response, 200
      
api.add_resource(TariffPrice, "/tariff-prices/")

app.run(debug=True)