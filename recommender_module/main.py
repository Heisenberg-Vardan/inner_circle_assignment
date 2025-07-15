import pandas as pd
import os
import joblib

from src.trainer import UserRecommenderTrainer
from src.recommender import UserRecommender
from src.persistence import save_model, load_model

# Load CSVs
users_df = pd.read_csv("data/users.csv")
interactions_df = pd.read_csv("data/activity.csv")

# Load model if exists
model_path = "models/xgb_model.pkl"
preprocessor_path = "models/preprocessor.pkl"

if os.path.exists(model_path) and os.path.exists(preprocessor_path):
    print("✅ Loading existing model...")
    model, preprocessor = load_model(model_path, preprocessor_path)
else:
    print("🧠 No model found. Starting training...")
    trainer = UserRecommenderTrainer(users_df, interactions_df)
    model, preprocessor = trainer.train()
    save_model(model, preprocessor, model_path, preprocessor_path)

# Instantiate recommender
recommender = UserRecommender(users_df, interactions_df, model, preprocessor)

# Example recommendation
user_id_to_recommend = users_df['user_id'].iloc[2]
recommendations = recommender.recommend_users(user_id_to_recommend, k=5)

print(f"\n📌 Top 5 recommendations for user {user_id_to_recommend}:")
for i, (uid, reason) in enumerate(recommendations):
    print(f"{i+1}. User {uid} - {reason}")

