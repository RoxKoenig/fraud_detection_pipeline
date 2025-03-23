import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Old (machine-specific path):
# HISTORICAL_DATA_FOLDER = os.path.expanduser("~/SynologyDrive/Uni/Backups/Python Code/fraud-detection")

# ✅ New (CI-friendly relative path):
HISTORICAL_DATA_FOLDER = os.path.join(os.getcwd(), "data")


def ensure_folder_exists(folder_path):
    """
    Ensure the folder exists, and if not, create it.
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logging.info(f"Folder created: {folder_path}")
        else:
            logging.info(f"Folder already exists: {folder_path}")
    except PermissionError as e:
        logging.error(f"Permission denied while creating folder {folder_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error ensuring folder exists {folder_path}: {e}")
        raise

def save_historical_data(data, file_name="historical_data.csv"):
    """
    Save historical data to a CSV file.
    """
    try:
        ensure_folder_exists(HISTORICAL_DATA_FOLDER)
        file_path = os.path.join(HISTORICAL_DATA_FOLDER, file_name)
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data to save is not a valid Pandas DataFrame.")
        data.to_csv(file_path, index=False)
        logging.info(f"Historical data saved successfully to {file_path}.")
    except ValueError as ve:
        logging.error(f"Value error while saving historical data: {ve}")
    except PermissionError as e:
        logging.error(f"Permission denied while saving data to {file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while saving historical data: {e}")

def load_historical_data(file_name="historical_data.csv"):
    """
    Load historical data from a CSV file.
    """
    file_path = os.path.join(HISTORICAL_DATA_FOLDER, file_name)
    try:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            logging.info(f"Historical data loaded successfully from {file_path}.")
            return data
        else:
            logging.warning(f"No historical data found at {file_path}.")
            return None
    except pd.errors.EmptyDataError:
        logging.error(f"The file {file_path} is empty or corrupted.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while loading historical data from {file_path}: {e}")
        return None

def detect_data_drift(historical_data, new_data, threshold=0.05):
    """
    Detect data drift by comparing historical and new distributions.
    """
    if historical_data is None:
        logging.warning("No historical data available. Drift detection skipped.")
        return False

    try:
        # Normalize distributions
        historical_dist = historical_data.set_index('fraud_label')['count'] / historical_data['count'].sum()
        new_dist = new_data.set_index('fraud_label')['count'] / new_data['count'].sum()

        # Calculate drift metrics
        drift_metrics = abs(historical_dist - new_dist).fillna(0)
        logging.info(f"Drift metrics: {drift_metrics}")

        # Check for drift exceeding threshold
        if (drift_metrics > threshold).any():
            logging.info("Data drift detected.")
            return True
        else:
            logging.info("No significant data drift detected.")
            return False
    except KeyError as e:
        logging.error(f"Key error in detecting data drift: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in detecting data drift: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example new data with drift
    new_data = pd.DataFrame({
        'fraud_label': ['Fraud', 'Not Fraud'],
        'count': [129858, 70142]
    })

    # Load historical data
    historical_data = load_historical_data()

    # Check if historical data exists
    if historical_data is None:
        logging.info("No historical data found. Saving new data as the initial historical dataset.")
        save_historical_data(new_data)
    else:
        # Detect data drift
        drift_detected = detect_data_drift(historical_data, new_data)
        if drift_detected:
            logging.info("Drift detected. Saving new data as historical.")
            save_historical_data(new_data)
        else:
            logging.info("No drift detected. Historical data remains unchanged.")
