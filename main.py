import os
import pandas as pd
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from data_drift import load_historical_data, detect_data_drift, save_historical_data

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

# Define the query to fetch data
query = """
SELECT 
    p.fraud_label AS fraud_label,  
    f.income_details, 
    f.assets, 
    f.debts, 
    e.employment_status, 
    h.number_of_dependents, 
    h.combined_household_income, 
    s.proof_of_eligibility
FROM personal_information p
LEFT JOIN financial_information f ON p.id = f.personal_info_id
LEFT JOIN employment_information e ON p.id = e.personal_info_id
LEFT JOIN household_information h ON p.id = h.personal_info_id
LEFT JOIN supporting_documents s ON p.id = s.personal_info_id;
"""

def fetch_data(query, db_config):
    """
    Fetch data from the PostgreSQL database using SQLAlchemy.
    """
    try:
        from sqlalchemy import create_engine
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(db_url)
        df = pd.read_sql_query(query, engine)
        logging.info(f"Data fetched successfully. Retrieved {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

def build_pipeline(categorical_columns, numerical_columns):
    """
    Build a pipeline with preprocessing and model steps.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=42, max_iter=500, class_weight='balanced'))
    ])

    return pipeline

def train_and_save_pipeline(X, y):
    """
    Train the pipeline and save it to a file.
    """
    model_name = "LogisticRegression"

    with mlflow.start_run(run_name=model_name):
        logging.info(f"Training {model_name}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

        logging.info(f"Categorical columns: {categorical_columns}")
        logging.info(f"Numerical columns: {numerical_columns}")

        pipeline = build_pipeline(categorical_columns, numerical_columns)

        # SMOTE to balance classes
        X_train_preprocessed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
        logging.info(f"SMOTE resampled data shape: {X_train_resampled.shape}")

        # Fit model
        pipeline.named_steps['model'].fit(X_train_resampled, y_train_resampled)

        # Evaluate
        X_test_preprocessed = pipeline.named_steps['preprocessor'].transform(X_test)
        y_pred = pipeline.named_steps['model'].predict(X_test_preprocessed)
        y_pred_proba = pipeline.named_steps['model'].predict_proba(X_test_preprocessed)[:, 1]

        cr = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logging.info(f"ROC-AUC Score: {roc_auc:.4f}")
        logging.info("Classification Report:")
        logging.info(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", cr["accuracy"])
        mlflow.log_metric("precision", cr["1"]["precision"])
        mlflow.log_metric("recall", cr["1"]["recall"])
        mlflow.log_metric("f1_score", cr["1"]["f1-score"])
        mlflow.log_metric("roc_auc", roc_auc)

        # Save pipeline
        joblib.dump(pipeline, "fraud_detection_model.pkl")
        logging.info("Pipeline saved successfully.")

if __name__ == "__main__":
    logging.info("Starting the fraud detection pipeline.")
    try:
        historical_data = load_historical_data()
        new_data = fetch_data(query, db_config)

        fraud_label_dist = new_data['fraud_label'].replace({
            'Betrug': 'Fraud', 'Kein Betrug': 'Not Fraud'
        }).value_counts(normalize=True).reset_index(name='count')
        fraud_label_dist.columns = ['fraud_label', 'count']

        drift_detected = detect_data_drift(historical_data, fraud_label_dist, threshold=0.05)

        if drift_detected:
            logging.info("Data drift detected. Retraining the model...")
            X = new_data.drop(columns=['fraud_label'])
            y = new_data['fraud_label'].replace({'Betrug': 1, 'Kein Betrug': 0})
            train_and_save_pipeline(X, y)
            save_historical_data(fraud_label_dist)
        else:
            logging.info("No data drift detected. Skipping retraining.")
            if historical_data is None:
                save_historical_data(fraud_label_dist)

    except Exception as e:
        logging.error(f"Pipeline terminated due to an error: {e}")
