# Credit Card Customer Segmentation & Classification

## Overview

This project uses Machine Learning to segment credit card customers based on their financial behavior and predict their customer segment using supervised classification models.

The project combines unsupervised learning, supervised learning, model optimization, experiment tracking, and an interactive Streamlit application.

## Objectives

- Clean and preprocess customer financial data.
- Segment customers using clustering algorithms.
- Compare K-Means and DBSCAN.
- Train and optimize classification models to predict customer segments.
- Track experiments using MLflow.
- Save the best trained pipeline.
- Deploy an interactive prediction interface using Streamlit.

## Dataset

The dataset contains approximately 9,000 credit card customers and their financial activity over six months.

Features include:

- `BALANCE`
- `PURCHASES`
- `ONEOFF_PURCHASES`
- `INSTALLMENTS_PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`
- `PAYMENTS`

The customer identifier (`CUST_ID`) is removed during preprocessing, and rows containing missing values are excluded.

Place the original dataset in `data/raw/dataset.csv`.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- MLflow
- Joblib
- Streamlit
- uv

## Project Structure

```text
customer-segmentation-classification/
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       ├── df_clean.csv
│       └── df_prepare_clustering.csv
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── pipeline.py
├── models/
│   └── credit_card_customer_segmentation_pipeline.joblib
├── app/
│   └── app.py
├── dags/
├── mlruns/
├── mlflow.db
├── pyproject.toml
├── uv.lock
└── README.md
```

## Methodology

### 1. Data Preprocessing

- Load the original dataset.
- Remove `CUST_ID`.
- Handle missing values by removing incomplete rows.
- Apply `log1p` to reduce feature skewness.
- Standardize numerical features.

### 2. Customer Segmentation

PCA is used to reduce dimensionality before clustering.

Two clustering algorithms are evaluated:

- **K-Means:** The final configuration uses 2 PCA components and 3 clusters.
- **DBSCAN:** Evaluated with different PCA dimensions, `eps`, and `min_samples` values.

K-Means is selected for the final segmentation based on the evaluated clustering metrics.

### 3. Customer Classification

The generated customer segments are used as target labels for supervised learning.

Four classification models are compared:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)

GridSearchCV with 5-fold cross-validation is used to optimize model hyperparameters.

### 4. Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The best model is selected based on its F1-score on the test set.

### 5. MLflow Tracking

MLflow records model parameters, evaluation metrics, and trained model artifacts to support experiment comparison and reproducibility.

### 6. Streamlit Application

The application allows users to enter customer financial information and predict the corresponding customer segment using the saved classification pipeline.

## Results

The optimized Random Forest model achieved the following results on the test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 96.54% |
| Precision | 96.57% |
| Recall    | 96.54% |
| F1-score  | 96.54% |

**Best hyperparameters:**

- `n_estimators`: 300
- `max_depth`: 20
- `min_samples_split`: 5

These results reflect performance on the generated cluster labels; they do not represent independently verified real-world customer categories.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <https://github.com/ISMAILBENMEZI/customer-segmentation-classification.git>
cd customer-segmentation-classification
```

Install the dependencies using uv:

```bash
uv sync
```

Ensure the dataset is available at `data/raw/dataset.csv`.

## Run the Pipeline

Execute the complete preprocessing, clustering, classification, optimization, and model-saving workflow:

```bash
uv run python -m src.pipeline
```

The trained pipeline is saved to:

```text
models/credit_card_customer_segmentation_pipeline.joblib
```

## Run the Streamlit Application

```bash
uv run streamlit run app/app.py
```

Open the local URL displayed in the terminal.

## View MLflow Experiments

Start the MLflow interface:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

http://127.0.0.1:5000

## Future Improvements

- Automate the workflow with Apache Airflow.
- Add more detailed cluster profiling and visualizations.
- Improve monitoring and experiment comparison.
- Deploy the Streamlit application.

## Author

**Ismail Benmezi**

GitHub: [ISMAILBENMEZI](https://github.com/ISMAILBENMEZI)
