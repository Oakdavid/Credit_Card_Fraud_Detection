from repositories.data_loader import FraudRepository
from services.fraud_service import FraudService

repo = FraudRepository()

service = FraudService(repo)

# service.inspect_data()

x_train, x_test, y_train, y_test = service.prepare_data()

model, metrics = service.train_model(
    x_train,
    y_train,
    x_test,
    y_test
)

print("Model Metrics:\n", metrics)
print("roc_auc_score:", metrics['roc_auc'])

best_model, best_metrics = service.select_best_model(
    x_train,
    y_train,
    x_test,
    y_test
)

print("Best Model Metrics:\n", best_metrics)