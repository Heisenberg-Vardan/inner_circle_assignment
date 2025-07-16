from utils import get_env_var, load_csvs_to_db, run_sql_query_from_file, save_df_to_csv

def main():
    """
    Main orchestration function. Loads data, runs SQL aggregation, and saves output.
    """
    users_csv = get_env_var("USERS_PATH")
    interactions_csv = get_env_var("INTERACTIONS_PATH")
    db_path = get_env_var("DB_PATH")
    sql_file = get_env_var("AGG_SQL_PATH")
    output_csv = get_env_var("OUTPUT_CSV")

    conn = load_csvs_to_db(users_csv, interactions_csv, db_path)
    aggregated_df = run_sql_query_from_file(conn, sql_file)
    print(aggregated_df)
    save_df_to_csv(aggregated_df, output_csv)
    print(f"\nAggregation results saved to {output_csv}")

if __name__ == "__main__":
    main()
