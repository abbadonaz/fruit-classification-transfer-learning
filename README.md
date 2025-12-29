# 🍎🥭 Fruit Classification with Transfer Learning

## Overview

This project demonstrates how to classify fruit images using transfer learning with a pre-trained VGG16 model. By leveraging ImageNet knowledge and fine-tuning custom layers, the model learns to classify a smaller fruit dataset efficiently with limited computational resources.

## 🎯 Aim

Train a convolutional neural network (CNN) that can classify fruit images into their respective categories.

## 🔑 Key Takeaways

Transfer Learning lets us reuse pre-trained models for new tasks with smaller datasets.

Data Augmentation improves generalization and helps prevent overfitting.

Fine-tuning specific layers yields better performance than feature extraction alone.

## 📥 Dataset Setup

This project uses the Fruit-Images Dataset
 from Kaggle.

1. Enable Kaggle API

Go to your Kaggle Account Settings:

- Scroll to API → click "Create New API Token".
This downloads a file called kaggle.json.

- Place kaggle.json in the correct location:


Linux/Mac: ~/.kaggle/kaggle.json
Windows: C:\Users\<YourUser>\.kaggle\kaggle.json

Ensure correct permissions (Linux/Mac only):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

2. Install Kaggle CLI

```bash
pip install kaggle

```
3. Download and Prepare Data

Run the setup script from the project root:
```bash
python src/get_data.py
```

This will:

- ⬇️ Download the dataset from Kaggle

- 📂 Extract files into data/fruits/

- 🔀 Split the training set into train/ and valid/

Final structure will look like:

data/fruits/
├── train/
│   ├── Apple/
│   ├── Banana/
│   └── ...
├── valid/
│   ├── Apple/
│   ├── Banana/
│   └── ...
└── test/
    ├── Apple/
    ├── Banana/
    └── ...


✅ After this step, you’re ready to train your model with:

from src.train import train_model
model, history, test_gen = train_model("data/fruits/train", "data/fruits/valid", "data/fruits/test")


## ✅ Final Output

A trained VGG16-based CNN capable of classifying fruit images into multiple categories with high accuracy.
