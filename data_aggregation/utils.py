import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(key, default=None):
    """
    Retrieve an environment variable's value or a default.

    Args:
        key (str): Environment variable name.
        default: Default value if the environment variable is not set.

    Returns:
        str: The value of the environment variable, or the default.

    Raises:
        ValueError: If the environment variable is not set and no default is provided.
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not set and no default provided.")
    return value

def ensure_parent_dir_exists(filepath):
    """
    Ensure the parent directory of the given file path exists. Creates it if needed.

    Args:
        filepath (str): The file path whose parent directory should exist.
    """
    dirpath = os.path.dirname(os.path.abspath(filepath))
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def load_csvs_to_db(users_csv, interactions_csv, db_path):
    """
    Loads users and interactions CSVs into a SQLite database as 'users' and 'interactions' tables.

    Args:
        users_csv (str): Path to users CSV file.
        interactions_csv (str): Path to interactions CSV file.
        db_path (str): Path for SQLite database file.

    Returns:
        sqlite3.Connection: Connection to the SQLite database.
    """
    users = pd.read_csv(users_csv)
    interactions = pd.read_csv(interactions_csv)
    ensure_parent_dir_exists(db_path)
    conn = sqlite3.connect(db_path)
    users.to_sql('users', conn, if_exists='replace', index=False)
    interactions.to_sql('interactions', conn, if_exists='replace', index=False)
    return conn

def run_sql_query_from_file(conn, sql_file_path):
    """
    Executes a SQL query loaded from a file and returns the results as a DataFrame.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        sql_file_path (str): Path to the SQL query file.

    Returns:
        pd.DataFrame: Query result as a DataFrame.
    """
    with open(sql_file_path, 'r') as f:
        query = f.read()
    return pd.read_sql_query(query, conn)

def save_df_to_csv(df, output_csv):
    """
    Saves a DataFrame to a CSV file, creating parent directories if needed.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_csv (str): Path to the output CSV file.
    """
    ensure_parent_dir_exists(output_csv)
    df.to_csv(output_csv, index=False)
