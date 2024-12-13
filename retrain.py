import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud_detection_experiment")

# Database configuration
db_config = {
    "dbname": "fraud_detection_db",
    "user": "admin",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}

def fetch_new_data(month):
    """
    Fetch new data for a specific month from the PostgreSQL database.
    """
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    query = f"""
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
    data = pd.read_sql(query, engine)
    if data.empty:
        raise ValueError(f"No data found for month {month}.")
    return data

def preprocess_data(data):
    """
    Preprocess the data for model training and evaluation.
    """
    # Handle missing values
    data = data.fillna(0)

    # Extract date features
    data['day_of_week'] = pd.to_datetime(data['transaction_date']).dt.dayofweek
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Drop `transaction_date`
    target = 'fraud_label'
    features = [col for col in data.columns if col not in [target, 'transaction_date']]

    # Separate features and target
    X = data[features]
    y = data[target]

    # Handle categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

def log_confusion_matrix(cm, labels, run_name):
    """
    Log confusion matrix as an artifact.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    artifact_path = f"{run_name}_confusion_matrix.png"
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)

def retrain_model(month):
    """
    Retrain the fraud detection model on new data for the specified month.
    """
    try:
        # Fetch and preprocess data
        print(f"Fetching data for month {month}...")
        data = fetch_new_data(month)
        X_train, X_test, y_train, y_test = preprocess_data(data)
    except Exception as e:
        print(f"Error fetching or preprocessing data: {e}")
        return

    # Train a logistic regression model
    print("Training the model...")
    model = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Log to MLflow
    print("Logging the model to MLflow...")
    try:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "Logistic Regression")
            mlflow.log_param("month_trained", month)
            mlflow.log_param("class_weight", "balanced")

            # Log metrics
            mlflow.log_metric("roc_auc", roc_auc)

            # Log confusion matrix as an artifact
            log_confusion_matrix(cm, labels=["Not Fraud", "Fraud"], run_name="LogisticRegression")

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="fraud_detection_model",
                signature=mlflow.models.signature.infer_signature(pd.DataFrame(X_train), y_train),
            )
            print(f"Model logged successfully to MLflow with ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"Error logging the model to MLflow: {e}")

if __name__ == "__main__":
    # Specify the month to train on
    retrain_month = 1  # January
    retrain_model(retrain_month)
