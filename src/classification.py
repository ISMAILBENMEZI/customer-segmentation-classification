from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
import mlflow.sklearn

def prepare_classification_data(df_clean, num_colu):
    X = df_clean[num_colu].copy()
    y = df_clean["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_classification_models():
    models = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
    }

    return models


def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }

        trained_models[name] = model

    return trained_models, results


def optimize_models(models, X_train, y_train):
    param_grid = {
        "Logistic Regression": {"model__C": [0.1, 1, 10, 100]},
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "Decision Tree": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "SVM": {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        },
    }

    best_models = {}
    best_params = {}

    for name, model in models.items():
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
    return best_models, best_params


def evaluate_best_models(best_models, X_test, y_test):
    results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(
                y_test,
                y_pred,
            ),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }

    return results


def select_best_model(best_models, results):
    best_name = max(results, key=lambda name: results[name]["f1"])

    best_model = best_models[best_name]

    return best_name, best_model


def save_best_model(best_model):
    joblib.dump(
        best_model,
        "models/credit_card_customer_segmentation_pipeline.joblib"
    )

def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("classification_clients")

def log_model_experiment(name , model , params , metrics):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )