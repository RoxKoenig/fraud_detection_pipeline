import argparse
import pandas as pd
from sqlalchemy import create_engine

def fetch_monthly_data(month, output_path=None):
    """
    Fetches data from the PostgreSQL database for a specified month and optionally saves it to a CSV file.

    Args:
        month (int): The month (1-12) for which to fetch data.
        output_path (str, optional): Path to save the fetched data as a CSV file. Defaults to None.
    """
    # Adjust database connection string to your setup
    engine = create_engine('postgresql://admin:password@localhost:5432/fraud_detection_db')

    # Generate SQL query to fetch data for the specified month
    query = f"""
        SELECT *
        FROM (
            SELECT 
                transaction_id, 
                transaction_date, 
                amount, 
                customer_id, 
                fraud_flag, 
                EXTRACT(MONTH FROM transaction_date) AS transaction_month
            FROM transactions
        ) subquery
        WHERE transaction_month = {month};
    """

    print(f"Fetching data for month {month}...")
    data = pd.read_sql(query, engine)

    if data.empty:
        print(f"No data found for month {month}.")
        return None

    print(f"Fetched {len(data)} records for month {month}.")
    
    # Save to CSV if output_path is specified
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Data for month {month} saved to {output_path}.")

    return data

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fetch monthly data from the PostgreSQL database.")
    parser.add_argument("--month", type=int, required=True, help="The month (1-12) for which to fetch data.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the fetched data as a CSV file.")
    args = parser.parse_args()

    # Fetch data for the specified month
    fetch_monthly_data(month=args.month, output_path=args.output_path)
