import pandas as pd
import sqlite3
import os

def load_data_to_db(users_csv_path, interactions_csv_path, db_path):
    """
    Loads user and interaction data from CSV files into a SQLite database.
    """
    try:
        users_df = pd.read_csv(users_csv_path)
        interactions_df = pd.read_csv(interactions_csv_path)

        # Establish a connection to the SQLite database
        # This will create the DB at the specified path if it doesn't exist
        conn = sqlite3.connect(db_path)

        users_df.to_sql('users', conn, if_exists='replace', index=False)
        interactions_df.to_sql('interactions', conn, if_exists='replace', index=False)

        print(f"Data loaded successfully into '{db_path}'")
        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- FIX: Build all paths relative to the script's location ---
    data_dir = os.path.join(script_dir, '..', 'data') # Go up one level, then into 'data'
    db_file = os.path.join(script_dir, 'interactions.db') # DB inside the 'data_aggregation' folder
    users_csv = os.path.join(data_dir, 'users.csv')
    interactions_csv = os.path.join(data_dir, 'activity.csv')
    
    # Load the data into the database
    load_data_to_db(users_csv, interactions_csv, db_file)