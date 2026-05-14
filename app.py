from flask import Flask, request, jsonify

from repositories.data_loader import FraudRepository
from services.prediction_service import PredictionService
from services.training_service import TrainingService
from services.data_service import DataService

app = Flask(__name__)

repo = FraudRepository()

prediction_service = PredictionService(repo)


@app.route('/')
def home():
    return "Fraud Detection API Running"


@app.route('/predict', methods=['POST'])
def predict():

    try:

        transaction_data = request.get_json()

        if transaction_data is None:
            return jsonify({
                "error": "Invalid input: Please input a data"
            }), 400

        result = prediction_service.predict_transaction(
            transaction_data
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True)