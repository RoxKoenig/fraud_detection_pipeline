# 🔍 Fraud Detection MLOps Pipeline

A robust end-to-end MLOps pipeline for fraud detection. This project integrates **data ingestion**, **drift detection**, **model retraining**, and **deployment** — all automated using **Docker**, **PostgreSQL**, **MLflow**, and **GitHub Actions**.

---

## 📌 Project Overview

This project simulates a real-world fraud detection scenario with:

- 🏷️ **Labelled data generation** per month
- 🔄 **Data drift detection** to trigger retraining
- 🧠 **Model retraining & evaluation** using logistic regression
- 🚀 **Model deployment** via Gunicorn
- 💾 **PostgreSQL** for database storage and backups
- 🧪 **MLflow** for experiment tracking and model registry
- ⚙️ **GitHub Actions** for CI/CD automation

---

## 🛠️ Technologies Used

- Python 3.10+
- Scikit-learn
- MLflow
- PostgreSQL
- Docker & Docker Compose
- GitHub Actions
- Pandas, NumPy, SMOTE, psycopg2

---

## 📂 Project Structure

fraud-detection-mlops/ ├── app.py # API server (Gunicorn) ├── main.py # Main pipeline execution ├── data_drift.py # Drift detection logic ├── retrain.py # Model retraining pipeline ├── deploy_model.py # Deployment script ├── generate_monthly_data.py # Synthetic data generation ├── database.py # PostgreSQL connection logic ├── fraud_detection_model.pkl # Trained model (v1) ├── fraud_detection_model_v2.pkl # Updated model (v2) ├── fraud_detection_db_backup.sql # PostgreSQL DB dump ├── requirements.txt # Python dependencies ├── docker-compose.yml # Docker service orchestration ├── Dockerfile / mlflow.Dockerfile # Docker images ├── init-mlflow-db.sql # Initial MLflow DB setup ├── .github/workflows/ # GitHub Actions workflow files └── README.md # Project documentation

---

## 🚀 How to Run This Project Locally

### 🔧 Prerequisites

- Docker & Docker Compose
- Git
- Python 3.10+ (if running scripts manually)

---

### 🧪 Running the MLOps Pipeline

1. **Clone the Repository**

```bash
git clone https://github.com/rpok1/fraud-detection-mlops.git
cd fraud-detection-mlops

2. Start Services with Docker Compose
docker-compose up --build

This command will:

    Launch PostgreSQL, MLflow, and all supporting services

    Initialize the database schema

    Run the full fraud detection pipeline:

        Generate monthly data

        Detect drift

        Retrain and evaluate the model

        Deploy the trained model via Gunicorn

        Register the model in MLflow

💻 Accessing the Services

    MLflow Tracking UI → http://localhost:5001

    Fraud Detection API (Gunicorn) → http://localhost:5000

📈 Sample Logs (CI/CD)

Here’s what happens during an automated GitHub Actions run:

    Drift is detected on new monthly data

    Model is retrained using Logistic Regression

    ROC-AUC and classification metrics are printed

    The new model is deployed and tracked in MLflow

🧪 Model Evaluation Snapshot

ROC-AUC Score: 0.4975

Classification Report:
              precision    recall  f1-score   support
       0       0.51      0.93      0.66      2063
       1       0.47      0.06      0.11      1937
accuracy                           0.51      4000

🔄 GitHub Actions Automation

    Runs every month (1st of the month) and on manual dispatch

    Loads PostgreSQL backup and checks for drift

    Retrains and deploys models if drift is found

    Uploads updated historical data as an artifact

    📅 Cron schedule: 0 0 1 * *
    🧪 Also triggerable manually via GitHub Actions UI

🔮 Future Enhancements

    Add Prometheus + Grafana monitoring

    Integrate FastAPI for a robust API layer

    Add alert system (Slack/Email) when drift is detected

    Use Docker volumes for persistent storage

📄 References

    Scikit-learn documentation: https://scikit-learn.org

    MLflow Tracking: https://www.mlflow.org

    Timescale/PostgreSQL: https://www.postgresql.org

    O'Reilly (Kreps, 2014): Questioning the Lambda Architecture

👤 Maintainer

Rox Koenig
📬 GitHub Profile
