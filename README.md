# Credit Card Fraud Detection (Python)

This project builds and evaluates Decision Tree and Random Forest classifiers for credit card fraud detection. It includes:

- EDA: missing values, outliers, class imbalance, and visualizations
- 70/30 train-test split
- Baseline models: Decision Tree and Random Forest
- Handling class imbalance with SMOTE (from imbalanced-learn)
- **Advanced techniques**: Feature engineering, multiple resampling methods, improved model tuning
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrices
- **Model deployment**: NVIDIA Triton Inference Server integration

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

### Basic Training (Baseline + SMOTE)

Train Decision Tree and Random Forest on the original data (baseline) and with SMOTE. Metrics and artifacts are saved under `artifacts/`.

```bash
python src/train_models.py --data-path data/creditcard.csv --test-size 0.3 --random-state 42
```

### Advanced Training (Recommended)

Use the improved training script with advanced techniques for better fraud detection:

```bash
python src/improved_retraining.py --data-path data/creditcard.csv --deploy
```

**Key improvements in advanced training:**
- **Feature Engineering**: Creates 5 additional fraud-specific features
  - Sum of absolute V-values (overall magnitude)
  - Standard deviation of V-values (variability)
  - Maximum absolute V-value (extreme values)
  - Amount-to-time ratio (unusual timing patterns)
  - Number of outliers (V-features with abs value > 2)
- **Advanced Resampling**: Tests multiple methods (SMOTE, ADASYN, SMOTEENN, SMOTETomek) and selects the best
- **Better Model Configuration**: Optimized hyperparameters for fraud detection
- **Comprehensive Validation**: Tests with realistic fraud patterns
- **Automatic Deployment**: Optionally deploys to Triton Inference Server

**Artifacts from basic training:**
- `artifacts/metrics_baseline.csv` and `artifacts/metrics_smote.csv`
- `artifacts/plots/*_confusion_matrix.png` (baseline and SMOTE)
- `artifacts/plots/*_{roc,pr,score_distribution}.png` (baseline and SMOTE)
- `artifacts/models/*.{joblib}`

**Artifacts from advanced training:**
- `artifacts/models/random_forest_improved.joblib`
- `artifacts/improved_evaluation_results.csv`
- Enhanced model with 35 features (30 original + 5 engineered)

## 5) Model Deployment

The project includes deployment to NVIDIA Triton Inference Server for production-ready inference:

```bash
# See detailed deployment instructions
cat deploy/triton/README_TRITON.md
```

**Quick deployment steps:**
1. Train the improved model: `python src/improved_retraining.py --data-path data/creditcard.csv --deploy`
2. Start Triton server: See `deploy/triton/README_TRITON.md`
3. Test the deployed model: `python deploy/triton/test_fraud_detection.py`

## 6) Notes

- SMOTE is applied only to the training split to avoid leakage.
- Trees do not require feature scaling.
- The improved model uses 35 features (30 original + 5 engineered features).
- Advanced training automatically selects the best resampling method.

## 7) Requirements

See `requirements.txt` for the full list of Python packages.

