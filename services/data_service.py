from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataService:
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
        