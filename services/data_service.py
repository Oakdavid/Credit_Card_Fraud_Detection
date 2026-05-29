from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataService:
    def __init__(self, repository):
        self.repository = repository
        self.scaler = StandardScaler()

    def inspect_data(self):
        data = self.repository.load_data()

        print("Data Types:\n", data.dtypes)

        return data

    def prepare_data(self):
        data = self.repository.load_data()

        print("Missing values in dataset:\n", data.isnull().sum())
        print("Duplicate rows in dataset:", data.duplicated().sum())

        data = data.drop_duplicates()
        data = data.fillna(0)

        features = data.drop('Class', axis=1)
        target = data['Class']

        X = features.copy()

        X[['Time', 'Amount']] = self.scaler.fit_transform(
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

        print(f"Training set shape: {x_train.shape}")
        print(f"Testing set shape: {x_test.shape}")

        return x_train, x_test, y_train, y_test
    
    def get_sample_transaction(self):
        data = self.repository.load_data()

        sample = (
            data
            .drop(columns=['Class'])
            .sample(1)
            .to_dict(orient='records')[0]
        )

        return sample