import pandas as pd
from src.utils import get_preferred_genders
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class UserRecommender:
    """
    User recommender system for generating user-to-user recommendations based on historical interactions.
    """

    def __init__(self, users_df: pd.DataFrame, interactions_df: pd.DataFrame, model, preprocessor, tfidf):
        """
        Initialize the UserRecommender.

        Args:
            users_df (pd.DataFrame): DataFrame of user profiles.
            interactions_df (pd.DataFrame): DataFrame of user interactions.
            model: Trained classifier for recommendations.
            preprocessor: Fitted preprocessor/transformer for input features.
            tfidf: Fitted TfidfVectorizer for 'age_about_me' fields.
        """
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.model = model
        self.preprocessor = preprocessor
        self.tfidf = tfidf

    def summarize_user_history(self, user_id):
        """
        Summarize the user's openness in location, gender, and age preferences,
        and infer sexual orientation (hetero, homo, bi).
        Returns:
            dict with city_openness, orientation, age_preference
        """
        user = self.users_df[self.users_df["user_id"] == user_id].iloc[0]
        user_gender = user["gender"]
        user_interactions = self.interactions_df[self.interactions_df["user_id"] == user_id]
        liked_ids = user_interactions[user_interactions["like_type"].isin([1, 2])]["liked_user_id"]
        liked_users = self.users_df[self.users_df["user_id"].isin(liked_ids)]


        unique_cities = liked_users["city"].nunique()
        if unique_cities == 0:
            city_openness = "N/A"
        elif unique_cities == 1:
            city_openness = "Single location"
        else:
            city_openness = "Multiple locations"

        liked_genders = set(liked_users["gender"].dropna().unique())
        if len(liked_genders) == 0:
            orientation = "N/A"
        elif len(liked_genders) == 1:
            only_gender = list(liked_genders)[0]
            if only_gender == user_gender:
                orientation = "homo"
            else:
                orientation = "hetero"
        else:
            orientation = "bi"

        if not liked_users.empty:
            age_diff = (liked_users["age"] - user["age"]).abs()
            similar_age = (age_diff <= 10).sum()
            if similar_age / len(liked_users) > 0.5:
                age_preference = "Prefers similar age"
            else:
                age_preference = "Open to wide age range"
        else:
            age_preference = "N/A"

        return {
            "city_openness": city_openness,
            "orientation": orientation,
            "age_preference": age_preference
        }


    def recommend_users(self, user_id, k=5):
        user = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        other_users = self.users_df[self.users_df['user_id'] != user_id]

        age_difference = abs(user['age'] - other_users['age'])
        age_difference_flag = (age_difference <= 10).astype(int)
        same_city = (user['city'] == other_users['city']).astype(int)
        same_gender = (user['gender'] == other_users['gender']).astype(int)

        about_me_user = [user['about_me'] if pd.notnull(user['about_me']) else ""] * len(other_users)
        about_me_liked = other_users['about_me'].fillna("").tolist()
        tfidf = joblib.load("tfidf.joblib")
        emb_user = tfidf.transform(about_me_user)
        emb_liked = tfidf.transform(about_me_liked)
        about_me_similarity = cosine_similarity(emb_user, emb_liked).diagonal()

        user_pref_gender = get_preferred_genders(self.interactions_df, self.users_df)
        preferred_genders = user_pref_gender.get(user_id, set())
        matches_preferred_gender = other_users['gender'].apply(lambda g: int(g in preferred_genders)).values

        X_candidates = pd.DataFrame({
            'age_difference_flag': age_difference_flag,
            'same_city': same_city,
            'same_gender': same_gender,
            'about_me_similarity': about_me_similarity,
            'matches_preferred_gender': matches_preferred_gender
        }, index=other_users.index)

        X_transformed = self.preprocessor.transform(X_candidates)
        probs = self.model.predict_proba(X_transformed)[:, 1]
        X_candidates['probability'] = probs

        top_k = X_candidates.sort_values(by='probability', ascending=False).head(k)

        recommendations = []
        for idx, row in top_k.iterrows():
            recommended_user_id = other_users.loc[idx, 'user_id']
            explanation = (
                f"Predicted Like Probability: {row['probability']:.2f} | "
                f"Age diff ≤10: {row['age_difference_flag']}, Same city: {row['same_city']}, "
                f"Same gender: {row['same_gender']}, About similarity: {row['about_me_similarity']:.2f}, "
                f"Matches orientation: {row['matches_preferred_gender']}"
            )
            recommendations.append((int(recommended_user_id), explanation))
        return recommendations
