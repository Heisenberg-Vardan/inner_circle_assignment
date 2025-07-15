import sqlite3
import pandas as pd
import os # Import the 'os' module

def run_aggregation_query(db_path, query_path):
    """
    Connects to the database, runs the aggregation query, and returns the result.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Read the SQL query from the file
        with open(query_path, 'r') as f:
            query = f.read()

        # Execute the query and load the result into a pandas DataFrame
        aggregated_df = pd.read_sql_query(query, conn)

        # Close the connection
        conn.close()

        return aggregated_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    # --- FIX: Build paths relative to the script's location ---
    
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths for the database and query files
    db_file = os.path.join(script_dir, 'interactions.db')
    query_file = os.path.join(script_dir, 'aggregation_query.sql')

    # Run the aggregation
    result_df = run_aggregation_query(db_file, query_file)

    if result_df is not None:
        print("Aggregated Interaction Metrics:")
        print(result_df.to_string())