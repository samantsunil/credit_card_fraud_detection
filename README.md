# Credit Card Fraud Detection (Python)

This project builds and evaluates Decision Tree and Random Forest classifiers for credit card fraud detection. It includes:

- EDA: missing values, outliers, class imbalance, and visualizations
- 70/30 train-test split
- Baseline models: Decision Tree and Random Forest
- Handling class imbalance with SMOTE (from imbalanced-learn)
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrices

## 1) Setup

### Create and activate a virtual environment (Windows - Git Bash / PowerShell)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Data

Use the public credit card transactions dataset with fraud labels. If you already have the CSV, place it at `data/creditcard.csv`.

If you want to download from Kaggle (`mlg-ulb/creditcardfraud`), set up Kaggle credentials and run:

```bash
# requires Kaggle API configured (~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --force
unzip -o data/creditcardfraud.zip -d data/
```

Expected file after setup:

```
data/creditcard.csv
```

## 3) EDA

Run EDA; this will print summaries and save plots to `artifacts/eda/`.

```bash
python src/eda.py --data-path data/creditcard.csv
```

## 4) Train and Evaluate Models

Train Decision Tree and Random Forest on the original data (baseline) and with SMOTE. Metrics and artifacts are saved under `artifacts/`.

```bash
python src/train_models.py --data-path data/creditcard.csv --test-size 0.3 --random-state 42
```

Artifacts:

- `artifacts/metrics_baseline.csv` and `artifacts/metrics_smote.csv`
- `artifacts/plots/*_confusion_matrix.png` (baseline and SMOTE)
- `artifacts/plots/*_{roc,pr,score_distribution}.png` (baseline and SMOTE)
- `artifacts/models/*.{joblib}`

## 5) Notes

- SMOTE is applied only to the training split to avoid leakage.
- Trees do not require feature scaling.

## 6) Requirements

See `requirements.txt` for the full list of Python packages.

