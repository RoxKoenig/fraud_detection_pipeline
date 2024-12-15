import pandas as pd
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_data(query, db_config):
    """
    Fetch data from PostgreSQL database using SQLAlchemy.
    """
    try:
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(db_url)
        df = pd.read_sql_query(query, engine)
        logging.info(f"Data fetched successfully. Retrieved {len(df)} rows.")
        logging.info(f"Columns in the fetched data: {df.columns}")  # Log columns here
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise
