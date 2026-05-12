from pathlib import Path
import pandas as pd
import joblib
from typing import Optional, Any

class FraudRepository:
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the repository with base paths."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.data_path = self.base_dir / "data" / "creditcard.csv"
        self.model_path = self.base_dir / "models" / "fraud_model.pkl"

        # Ensure folders exist
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load credit card data from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return pd.read_csv(self.data_path)

    def save_model(self, model: Any) -> None:
        """Save the trained model to disk."""
        joblib.dump(model, self.model_path)
        print(f"Model saved at {self.model_path}")

    def load_model(self) -> Any:
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return joblib.load(self.model_path)
