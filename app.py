from repositories.data_loader import FraudRepository
from services.prediction_service import PredictionService
from services.training_service import TrainingService
from services.data_service import DataService

repo = FraudRepository()

service = TrainingService(repo)
data_service = DataService(repo)
prediction_service = PredictionService(repo)

sample_transaction = {
    "Time": 0.1,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536346,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}

result = prediction_service.predict_transaction(sample_transaction)

print(result)

# service.inspect_data()
# service.prepare_data()

# x_train, x_test, y_train, y_test = service.prepare_data()

# model, metrics = service.train_model(
#     x_train,
#     y_train,
#     x_test,
#     y_test
# )

# print("Model Metrics:\n", metrics)
# print("roc_auc_score:", metrics['roc_auc'])

# best_model, best_metrics = service.select_best_model(
#     x_train,
#     y_train,
#     x_test,
#     y_test
# )

# print("Best Model Metrics:\n", best_metrics)