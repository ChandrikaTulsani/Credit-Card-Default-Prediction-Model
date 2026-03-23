# Credit Card Default Prediction

A binary classification project that predicts whether a credit card client will default on their next payment using machine learning. Built on the UCI Credit Card Default dataset with full EDA, preprocessing, model training, cross-validation, and feature importance analysis.

---

## Overview

Credit card default is a major risk for financial institutions. This project builds and compares two classification models — a Decision Tree and Logistic Regression — to identify clients at risk of defaulting, helping banks take proactive risk management actions.

---

## Dataset

**`default of credit card clients.xls`** — UCI Machine Learning Repository dataset containing payment records of 30,000 credit card clients in Taiwan.

| Property | Value |
|---|---|
| Total Samples | 30,000 |
| Features | 25 (24 input + 1 target) |
| No Default (0) | ~77.9% |
| Default (1) | ~22.1% |
| Missing Values | None |

**Key features:** `LIMIT_BAL` (credit limit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`–`PAY_6` (repayment status), `BILL_AMT1`–`BILL_AMT6` (bill amounts), `PAY_AMT1`–`PAY_AMT6` (payment amounts).

---

## Project Structure

```
├── bootcamp_assignment_2.ipynb    # Main notebook
├── default of credit card clients.xls  # Dataset
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Class distribution bar chart — identifies class imbalance (~78% no default vs ~22% default)
- Histograms of key numerical features (`LIMIT_BAL`, `AGE`, `BILL_AMT1`, `PAY_AMT1`) by default status
- Count plots of categorical features (`SEX`, `EDUCATION`, `MARRIAGE`) vs default
- Correlation heatmap across all numerical features
- Box plots to identify outliers in key features

### 2. Data Preprocessing
- No missing values found across all 25 columns
- Categorical features (`SEX`, `EDUCATION`, `MARRIAGE`) one-hot encoded using `OneHotEncoder`
- Numerical features standardized using `StandardScaler`
- Final processed shape: 30,000 × 32 features
- 80/20 stratified train-test split (24,000 train / 6,000 test)

### 3. Model Training

**Decision Tree Classifier**
```python
DecisionTreeClassifier(random_state=42)
```

**Logistic Regression**
```python
LogisticRegression(random_state=42)
```

### 4. Cross-Validation
5-fold cross-validation applied to both models across 5 metrics: Accuracy, Precision, Recall, F1, and AUC-ROC.

### 5. Evaluation
Models compared using classification reports, confusion matrices, and AUC-ROC curves.

---

## Results

### Model Performance (Hold-out Test Set)

| Metric | Decision Tree | Logistic Regression |
|---|---|---|
| Accuracy | 72.6% | **80.7%** |
| Precision | 38.6% | **68.8%** |
| Recall | 40.5% | 23.6% |
| F1 Score | 39.6% | 35.1% |

### 5-Fold Cross-Validation Results

| Metric | Decision Tree | Logistic Regression |
|---|---|---|
| CV Accuracy | 68.5% ± 4.2% | **80.7% ± 0.5%** |
| CV Precision | 32.7% ± 5.0% | **69.0% ± 5.6%** |
| CV Recall | 38.4% ± 6.0% | 24.9% ± 7.8% |
| CV F1 | 35.0% ± 4.7% | 35.5% ± 7.8% |
| **CV AUC-ROC** | 0.577 ± 0.034 | **0.719 ± 0.017** |

> Logistic Regression is the stronger overall model with higher accuracy, precision, and AUC-ROC. The Decision Tree has a slight edge in recall, meaning it catches more actual defaults but at the cost of more false positives.

### Confusion Matrices

| | Decision Tree | Logistic Regression |
|---|---|---|
| True Negatives | 3,818 | 4,531 |
| False Positives | 855 | 142 |
| False Negatives | 789 | 1,014 |
| True Positives | 538 | 313 |

---

## Feature Importance (Top 10)

From the Decision Tree:

| Feature | Importance |
|---|---|
| `PAY_0` (repayment status) | 15.97% |
| `BILL_AMT1` | 6.06% |
| `AGE` | 6.02% |
| `LIMIT_BAL` | 5.51% |
| `PAY_AMT3` | 4.94% |
| `PAY_AMT2` | 4.85% |
| `PAY_AMT6` | 4.59% |
| `PAY_AMT1` | 4.36% |
| `PAY_AMT5` | 4.22% |

> Key insight: Repayment status (`PAY_0`) is the single strongest predictor of default, followed by bill amounts, age, and credit limit.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

## Usage

1. Place `default of credit card clients.xls` in the `/content/` directory (or update the file path).
2. Open `bootcamp_assignment_2.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells sequentially.

---

## Technologies Used

- **Python 3**
- **scikit-learn** — preprocessing, Decision Tree, Logistic Regression, cross-validation, evaluation metrics
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — EDA and result visualization
