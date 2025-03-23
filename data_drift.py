import os
import sys
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
HISTORICAL_DATA_FOLDER = os.path.join(os.getcwd(), "data")
HISTORICAL_DATA_FILE = os.path.join(HISTORICAL_DATA_FOLDER, "historical_data.csv")
NEW_DATA_FILE = os.path.join(os.getcwd(), "output.csv")

def ensure_folder_exists(folder_path):
    """Ensure the folder exists, and if not, create it."""
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Ensured folder exists: {folder_path}")

def save_historical_data(data):
    """Save historical data to CSV."""
    ensure_folder_exists(HISTORICAL_DATA_FOLDER)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data to save is not a valid Pandas DataFrame.")
    data.to_csv(HISTORICAL_DATA_FILE, index=False)
    logging.info(f"Historical data saved to {HISTORICAL_DATA_FILE}")

def load_historical_data():
    """Load historical data from CSV."""
    if os.path.exists(HISTORICAL_DATA_FILE):
        try:
            data = pd.read_csv(HISTORICAL_DATA_FILE)
            logging.info(f"Loaded historical data from {HISTORICAL_DATA_FILE}")
            return data
        except pd.errors.EmptyDataError:
            logging.warning(f"{HISTORICAL_DATA_FILE} is empty or corrupted.")
    else:
        logging.warning(f"No historical data found at {HISTORICAL_DATA_FILE}")
    return None

def detect_data_drift(historical_data, new_data, threshold=0.05):
    """
    Detect data drift by comparing the distribution of fraud labels.
    Returns True if drift is detected, False otherwise.
    """
    if historical_data is None or new_data is None:
        logging.warning("Insufficient data to detect drift.")
        return False

    try:
        hist_counts = historical_data['fraud_flag'].value_counts(normalize=True)
        new_counts = new_data['fraud_flag'].value_counts(normalize=True)

        all_labels = set(hist_counts.index).union(set(new_counts.index))
        hist_dist = hist_counts.reindex(all_labels, fill_value=0)
        new_dist = new_counts.reindex(all_labels, fill_value=0)

        drift_metrics = (hist_dist - new_dist).abs()
        logging.info(f"Drift metrics:\n{drift_metrics}")

        drift = (drift_metrics > threshold).any()
        if drift:
            logging.info("⚠️ Data drift detected.")
        else:
            logging.info("✅ No significant data drift detected.")

        return drift

    except KeyError as ke:
        logging.error(f"Missing expected column during drift detection: {ke}")
        return False
    except Exception as e:
        logging.error(f"Error during drift detection: {e}")
        return False

# ---------- MAIN EXECUTION BLOCK ----------

if __name__ == "__main__":
    # Load new data
    if not os.path.exists(NEW_DATA_FILE):
        logging.error(f"New data file {NEW_DATA_FILE} not found.")
        sys.exit(1)

    try:
        new_data = pd.read_csv(NEW_DATA_FILE)
        logging.info(f"New data loaded from {NEW_DATA_FILE}, shape: {new_data.shape}")

        # ✅ Fix column name inconsistency
        if 'fraud_flag' not in new_data.columns and 'fraud_label' in new_data.columns:
            new_data.rename(columns={'fraud_label': 'fraud_flag'}, inplace=True)
            logging.info("Renamed 'fraud_label' column to 'fraud_flag'.")

    except Exception as e:
        logging.error(f"Failed to load new data: {e}")
        sys.exit(1)

    historical_data = load_historical_data()

    if historical_data is None:
        logging.info("No historical data found. Saving new data as baseline.")
        save_historical_data(new_data)
        sys.exit(0)

    drift = detect_data_drift(historical_data, new_data)

    if drift:
        logging.info("Updating historical data due to detected drift.")
        save_historical_data(new_data)
        sys.exit(10)  # Exit code for drift detected
    else:
        logging.info("Keeping existing historical data. No update needed.")
        sys.exit(0)
