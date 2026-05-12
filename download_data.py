import kagglehub
import shutil
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Dataset downloaded to:", path)

# Optional: move CSV into local data folder
source = Path(path) / "creditcard.csv"
destination = Path("data/creditcard.csv")

destination.parent.mkdir(exist_ok=True)

shutil.copy(source, destination)

print("Dataset copied to:", destination)