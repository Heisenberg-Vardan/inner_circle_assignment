import pandas as pd

class UserRecommender:
    def __init__(self, users_df, interactions_df, model, preprocessor):
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.model = model
        self.preprocessor = preprocessor

    def summarize_user_history(self, user_id):
        user_interactions = self.interactions_df[self.interactions_df["user_id"] == user_id]
        summary = {
            "likes": len(user_interactions[user_interactions["like_type"] == 1]),
            "dislikes": len(user_interactions[user_interactions["like_type"] == 0]),
            "matches": len(user_interactions[user_interactions["like_type"] == 2]),
        }

        liked_ids = user_interactions[user_interactions["like_type"].isin([1, 2])]["liked_user_id"]
        liked_users = self.users_df[self.users_df["user_id"].isin(liked_ids)]

        if not liked_users.empty:
            top_city = liked_users["city"].mode().iloc[0]
            top_gender = liked_users["gender"].mode().iloc[0]
        else:
            top_city = "N/A"
            top_gender = "N/A"

        return summary, top_city, top_gender

    def recommend_users(self, user_id, k=5):
        user = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        other_users = self.users_df[self.users_df['user_id'] != user_id]

        # Create candidate pairs
        X_candidates = pd.DataFrame({
            'user_id': user_id,
            'liked_user_id': other_users['user_id'],
            'age_user': user['age'],
            'age_liked': other_users['age'],
            'age_difference': abs(user['age'] - other_users['age']),
            'same_city': (user['city'] == other_users['city']).astype(int),
            'same_gender': (user['gender'] == other_users['gender']).astype(int),
            'gender_user': user['gender'],
            'gender_liked': other_users['gender'],
            'city_user': user['city'],
            'city_liked': other_users['city'],
            'about_me_user': user['about_me'] if pd.notnull(user['about_me']) else "",
            'about_me_liked': other_users['about_me'].fillna("")
        })


        summary, top_city, top_gender = self.summarize_user_history(user_id)
        print(f"📊 User {user_id} Interaction History:")
        print(f"✔️ Likes: {summary['likes']} | ❌ Dislikes: {summary['dislikes']} | ❤️ Matches: {summary['matches']}")
        print(f"🏙️ Frequently liked city: {top_city}")
        print(f"🚻 Preferred gender: {top_gender}\n")



        # Predict probabilities
        X_transformed = self.preprocessor.transform(X_candidates)
        probs = self.model.predict_proba(X_transformed)[:, 1]
        X_candidates['probability'] = probs

        top_k = X_candidates.sort_values(by='probability', ascending=False).head(k)

        insights = []
        for _, row in top_k.iterrows():
            insights.append((
                int(row['liked_user_id']),
                f"Similarity score: {row['probability']:.2f} | Same city: {bool(row['same_city'])}, Same gender: {bool(row['same_gender'])}"
            ))

        return insights
    


