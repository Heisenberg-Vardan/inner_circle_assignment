import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(key: str, default=None):
    """
    Retrieve an environment variable with an optional default.
    Raises ValueError if not found and no default is provided.
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable '{key}' not found and no default provided.")
    return value


def load_data(users_path: str, interactions_path: str):
    """Load user and interaction data from CSV files."""
    users_df = pd.read_csv(users_path)
    interactions_df = pd.read_csv(interactions_path)
    return users_df, interactions_df

def save_model(model, preprocessor, model_path: str, preprocessor_path: str):
    """
    Save the trained model and preprocessor to disk.

    Args:
        model: Trained machine learning model.
        preprocessor: Preprocessing pipeline or transformer.
        model_path (str): Path to save the model file.
        preprocessor_path (str): Path to save the preprocessor file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print("💾 Model and preprocessor saved.")

def load_model(model_path: str, preprocessor_path: str):
    """
    Load the trained model and preprocessor from disk.

    Args:
        model_path (str): Path to the saved model file.
        preprocessor_path (str): Path to the saved preprocessor file.

    Returns:
        tuple: (model, preprocessor)
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def ensure_models(users_df, interactions_df, model_path, preprocessor_path, trainer_class):
    """
    Ensure model and preprocessor exist, loading them if possible or training and saving otherwise.
    """
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        print("✅ Loading existing model...")
        model, preprocessor = load_model(model_path, preprocessor_path)
    else:
        print("🧠 No model found. Starting training...")
        trainer = trainer_class(users_df, interactions_df)
        model, preprocessor = trainer.train()
        save_model(model, preprocessor, model_path, preprocessor_path)
    return model, preprocessor

def get_preferred_genders(interactions_df, users_df):
    """
    Returns a dict: user_id -> set of genders they have liked/matched with.
    """
    liked = interactions_df[interactions_df["like_type"].isin([1, 2])]
    liked = liked.merge(users_df, left_on="liked_user_id", right_on="user_id")
    return liked.groupby("user_id_x")["gender"].agg(lambda x: set(x)).to_dict()

def visualize_top_recommendations(users_df, recommendations, user_id, n_show=5):
    """
    Visualize the top N recommended users with their predicted probabilities
    and a comparison to the input user's main attributes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    top_recs = recommendations[:n_show]
    rec_ids = [uid for uid, _ in top_recs]
    probabilities = [float(info.split(':')[1].split('|')[0].strip()) for _, info in top_recs]

    user = users_df[users_df['user_id'] == user_id].iloc[0]
    rec_users = users_df[users_df['user_id'].isin(rec_ids)].copy()
    rec_users = rec_users.set_index('user_id').loc[rec_ids]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(range(n_show), probabilities, align='center', color='skyblue')
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f'User {uid}' for uid in rec_ids])
    ax.invert_yaxis()
    ax.set_xlabel('Predicted Like Probability')
    ax.set_title(f'Top {n_show} Recommendations for User {user_id} (Predicted Probability & Attribute Comparison)')

    for i, (uid, bar) in enumerate(zip(rec_ids, bars)):
        rec = rec_users.loc[uid]
        age_match = rec['age'] == user['age']
        city_match = rec['city'] == user['city']
        gender_match = rec['gender'] == user['gender']

        label = (
            f"Input: Age={user['age']}, City={user['city']}, Gender={user['gender']}\n"
            f"Rec.:  Age={rec['age']}, City={rec['city']}, Gender={rec['gender']}\n"
            f"City: {'MATCH' if city_match else 'NO MATCH' }, "
            f"Gender: {'MATCH' if gender_match else 'NO MATCH'}"
        )
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=9)
    plt.tight_layout()
    plt.show()
