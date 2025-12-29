import os
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def download_from_kaggle(dataset="moltean/fruits", output_dir = "data"):
    """
    Small helper function to download a dataset from Kaggle using the Kaggle API.

    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = Path(output_dir) / "fruits.zip"
    if not zip_path.exists():
        print("⬇️ Downloading dataset from Kaggle...")
        os.system(f"kaggle datasets download -d {dataset} -p {output_dir}")
        print("✅ Download complete.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print("📂 Data extracted to:", output_dir)

def prepare_data(output_dir="data/fruits", val_split=0.2):
    """
    Splits training data into train/valid sets.
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"

    if not train_dir.exists():
        raise RuntimeError("Train directory was not found. Did the dataset extract correctly?")

    for class_name in os.listdir(train_dir):
        class_path = train_dir / class_name
        if not class_path.is_dir():
            continue

        images = os.listdir(class_path)
        train_files, val_files = train_test_split(
            images, test_size=val_split, random_state=42
        )

        val_class_path = valid_dir / class_name
        val_class_path.mkdir(parents=True, exist_ok=True)

        for f in val_files:
            shutil.move(str(class_path / f), str(val_class_path / f))

    print("Validation split created.")


if __name__ == "__main__":
    download_from_kaggle(dataset="moltean/fruits", output_dir="data")
    #prepare_data(output_dir="data/fruits", val_split=0.2)