import pandas as pd


class PredictionService:
    def __init__(self, repository):
        self.repository = repository
        self.model = None

    def load_model(self):
        """
        Load the saved trained model
        """
        self.model = self.repository.load_model()
        print("Model loaded successfully.")

    def predict_transaction(self, transaction_data):
        """
        Predict fraud for a single transaction

        transaction_data:
            dictionary or DataFrame containing transaction values
        """

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Convert dictionary input to DataFrame
        if isinstance(transaction_data, dict):
            transaction_data = pd.DataFrame([transaction_data])

        # Predict class
        prediction = self.model.predict(transaction_data)[0]

        # Predict fraud probability
        probability = self.model.predict_proba(transaction_data)[0][1]

        result = {
            "prediction": int(prediction),
            "label": "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction",
            "fraud_probability": round(float(probability), 4)
        }

        return result