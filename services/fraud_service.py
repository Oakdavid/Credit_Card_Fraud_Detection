from pyexpat import features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


class FraudService:
    def __init__(self, repository):
        self.repository = repository
        self.model = None

    def inspect_data(self):
        data = self.repository.load_data()
        # print("First 5 rows of the dataset:\n", data.head())
        # print("Data Shape:", data.shape)
        print("Data Types:\n", data.dtypes)
        # print("Missing Values:\n", data.isnull().sum())
        # print("Class Distribution:\n", data['Class'].value_counts())

        return data
    
    
    def prepare_data(self):
        data = self.repository.load_data()

        print('Missing values in dataset:\n', data.isnull().sum())
        print('duplicate rows in dataset:', data.duplicated().sum())

        data = data.drop_duplicates()
        data = data.fillna(0)

        features = data.drop('Class', axis=1)
        target = data['Class']

        scaler = StandardScaler()

        X = features.copy()

        X[['Time', 'Amount']] = scaler.fit_transform(
            X[['Time', 'Amount']]
        )

        y = target.copy()

        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("Training set shape:", x_train.shape)
        print("Testing set shape:", x_test.shape)

        return x_train, x_test, y_train, y_test
        

    def train_model(self, x_train, y_train, x_test=None, y_test=None):
        # Train a Random Forest classifier and save the trained model
        if x_train is None or y_train is None:
            raise ValueError("x_train and y_train must be provided")

        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced', 
            n_jobs=-1)

        start = time.time()
        clf.fit(x_train, y_train)
        elapsed = time.time() - start

        self.model = clf
        print(f"Model trained in {elapsed:.2f}s")

        try:
            self.repository.save_model(self.model)
        except Exception as e:
            print("Warning: failed to save model:\n", e )
            print(f"Random forest trained in {elapsed:.2f}s but not saved to disk.")

        metrics = {}
        if x_test is not None and y_test is not None:
            y_pred = clf.predict(x_test)
            y_proba = clf.predict_proba(x_test)[:, 1]
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        return clf, metrics

   

    def select_best_model(self, x_train, y_train, x_test=None, y_test=None):
        """
        Trains multiple models and selects the best one based on ROC AUC.
        Returns the best model and its metrics.
        """
        # Define candidate models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        best_model_name = None
        best_model = None
        best_roc_auc = -1
        best_metrics = None

        for name, clf in models.items():
            print(f"Training {name}...")
            start = time.time()
            clf.fit(x_train, y_train)
            elapsed = time.time() - start
            print(f"{name} trained in {elapsed:.2f}s")

            if x_test is not None and y_test is not None:
                # Handle models without predict_proba
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(x_test)[:, 1]
                else:
                    y_proba = clf.decision_function(x_test)

                roc_auc = roc_auc_score(y_test, y_proba)
                y_pred = clf.predict(x_test)
                metrics = {
                    "roc_auc": roc_auc,
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred)
                }

                print(f"{name} ROC AUC: {roc_auc:.4f}")

                # Update best model
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_name = name
                    best_model = clf
                    best_metrics = metrics

        print(f"\nBest Model: {best_model_name} with ROC AUC: {best_roc_auc:.4f}")

        # Save best model
        try:
            self.repository.save_model(best_model)
            print(f"{best_model_name} saved to disk.")
        except Exception as e:
            print("Warning: failed to save model:\n", e)

        return best_model, best_metrics
    