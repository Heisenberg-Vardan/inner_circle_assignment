import argparse
from src.trainer import UserRecommenderTrainer
from src.recommender import UserRecommender
from src.utils import (
    load_data, ensure_models, visualize_top_recommendations, get_env_variable
)

def main():
    parser = argparse.ArgumentParser(description="User Recommendation Engine")
    parser.add_argument('--user_id', '-u', type=int, required=True, help='User ID for recommendations')
    parser.add_argument('--show', type=int, default=5, help='Number of recommendations to visualize (default: 5)')
    args = parser.parse_args()

    users_path = get_env_variable('USERS_PATH')
    interactions_path = get_env_variable('INTERACTIONS_PATH')
    model_path = get_env_variable('MODEL_PATH')
    preprocessor_path = get_env_variable('PREPROCESSOR_PATH')
    tfidf_path = get_env_variable('TFIDF_PATH', default='tfidf.joblib')

    users_df, interactions_df = load_data(users_path, interactions_path)
    model, preprocessor = ensure_models(
        users_df, interactions_df, model_path, preprocessor_path, UserRecommenderTrainer
    )

    import joblib
    tfidf = joblib.load(tfidf_path)

    recommender = UserRecommender(users_df, interactions_df, model, preprocessor, tfidf)

    summary = recommender.summarize_user_history(args.user_id)
    print("📍 City openness:", summary["city_openness"])
    print("⚧️ Orientation:", summary["orientation"])
    print("🎂 Age preference:", summary["age_preference"])


    recommendations = recommender.recommend_users(args.user_id, k=10)
    print(f"\n📌 Top {args.show} recommendations for user {args.user_id}:")
    for i, (uid, reason) in enumerate(recommendations[:args.show]):
        print(f"{i+1}. User {uid} - {reason}")

    visualize_top_recommendations(users_df, recommendations, args.user_id, n_show=args.show)

if __name__ == "__main__":
    main()
