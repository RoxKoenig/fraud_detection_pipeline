import os
import pandas as pd
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import mlflow

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MLflow Configuration
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud_detection_experiment")

# Database Configuration
db_config = {
    "dbname": "fraud_detection_db",
    "user": "admin",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}

# SQL query template to fetch data for a specific month
query_template = """
SELECT 
    t.transaction_date,
    t.amount,
    t.customer_id,
    t.fraud_flag AS fraud_label,
    f.income_details,
    f.assets,
    f.debts,
    e.employment_status
FROM transactions t
LEFT JOIN financial_information f ON t.customer_id = f.personal_info_id
LEFT JOIN employment_information e ON t.customer_id = e.personal_info_id
WHERE EXTRACT(MONTH FROM t.transaction_date) = {month};
"""

def fetch_new_data(month, db_config):
    """
    Fetch data for the given month from the database.
    """
    try:
        from sqlalchemy import create_engine
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(db_url)
        query = query_template.format(month=month)
        data = pd.read_sql_query(query, engine)
        logging.info(f"Data fetched successfully. Retrieved {len(data)} rows.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the data for model training.
    """
    # Handle missing values
    data.fillna(0, inplace=True)

    # Extract additional features
    data["day_of_week"] = pd.to_datetime(data["transaction_date"]).dt.dayofweek  # Corrected
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Define features and target
    target = "fraud_label"
    features = [col for col in data.columns if col not in [target, "transaction_date"]]
    X = data[features]
    y = data[target]

    # Define preprocessors
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    # Preprocess and handle class imbalance
    X_preprocessed = preprocessor.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

    return X_resampled, y_resampled

def train_and_register_model(X_train, y_train, X_test, y_test, month):
    """
    Train a Logistic Regression model and register it in MLflow Model Registry.
    """
    try:
        logging.info("Training the model...")
        model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logging.info(f"ROC-AUC: {roc_auc:.4f}")

        # Register the model in MLflow Model Registry
        with mlflow.start_run() as run:
            # Log model parameters and metrics
            mlflow.log_param("month_trained", month)
            mlflow.log_param("model_type", "Logistic Regression")
            mlflow.log_metric("roc_auc", roc_auc)

            # Register the model (without artifacts)
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/fraud_detection_model",
                name="fraud_detection_model"
            )
            logging.info("Model registered successfully to MLflow Model Registry.")

    except Exception as e:
        logging.error(f"Error during training or logging: {e}")
        raise

if __name__ == "__main__":
    retrain_month = 1  # Specify the month for retraining
    try:
        logging.info(f"Starting retraining pipeline for month: {retrain_month}")
        data = fetch_new_data(retrain_month, db_config)
        X_resampled, y_resampled = preprocess_data(data)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        # Train and register the model
        train_and_register_model(X_train, y_train, X_test, y_test, retrain_month)
    except Exception as e:
        logging.error(f"Retraining pipeline failed: {e}")
