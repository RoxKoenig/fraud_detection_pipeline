import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO)

def run_step(description, command):
    logging.info(f"🔄 {description}...")
    try:
        subprocess.run(command, check=True)
        logging.info(f"✅ {description} complete.\n")
    except subprocess.CalledProcessError:
        logging.error(f"❌ Failed: {description}\n")

# Simulate 12 months
for month in range(1, 13):
    logging.info(f"\n🚀 ======= Simulating Month {month} =======")

    # Step 1: Fetch monthly data from DB
    run_step(
        f"Fetching data for Month {month}",
        ["python", "generate_monthly_data.py", "--month", str(month), "--output_path", "output.csv"]
    )

    # Step 2: Run data drift detection
    run_step("Detecting data drift", ["python", "data_drift.py"])

    # Step 3: Run main pipeline logic (feature engineering, evaluation, etc.)
    run_step("Running main pipeline", ["python", "main.py"])

    # Step 4: Retrain model (if drift detected, your logic should handle this)
    run_step("Retraining model", ["python", "retrain.py"])

    # Step 5: Redeploy the model
    run_step("Deploying model as REST API", ["python", "deploy_model.py"])

    # (Optional) Add a short delay between months
    time.sleep(2)
