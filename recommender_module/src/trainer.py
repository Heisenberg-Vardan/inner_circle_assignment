from src.utils import get_preferred_genders
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

class UserRecommenderTrainer:
    """
    Trainer class for user recommender models.
    Handles feature engineering, model training, and evaluation.
    """

    def __init__(self, users_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Initialize the trainer.

        Args:
            users_df (pd.DataFrame): User profile data.
            interactions_df (pd.DataFrame): User interactions data.
        """
        self.users_df = users_df
        self.interactions_df = interactions_df
        self.user_pref_gender = get_preferred_genders(self.interactions_df, self.users_df)

    def create_training_data(self):
        """
        Merge data, engineer binary, similarity, and orientation features for model training.

        Returns:
            X (pd.DataFrame): Feature DataFrame for model training.
            y (pd.Series): Target variable indicating positive interaction.
        """
        df = self.interactions_df.merge(self.users_df, left_on='user_id', right_on='user_id')
        df = df.merge(self.users_df, left_on='liked_user_id', right_on='user_id', suffixes=('_user', '_liked'))

        df['age_difference'] = abs(df['age_user'] - df['age_liked'])
        df['age_difference_flag'] = (df['age_difference'] <= 10).astype(int)
        df['same_city'] = (df['city_user'] == df['city_liked']).astype(int)
        df['same_gender'] = (df['gender_user'] == df['gender_liked']).astype(int)
        df['about_me_user'] = df['about_me_user'].fillna("")
        df['about_me_liked'] = df['about_me_liked'].fillna("")
        df['target'] = df['like_type'].apply(lambda x: 1 if x in [1, 2] else 0)

        # Calculate matches_preferred_gender first (quickly!)
        df['matches_preferred_gender'] = df.apply(
            lambda row: int(row['gender_liked'] in self.user_pref_gender.get(row['user_id_user'], set())),
            axis=1
        )

        # Then do the expensive TF-IDF/cosine work:
        all_about = pd.concat([df['about_me_user'], df['about_me_liked']])
        tfidf = TfidfVectorizer(max_features=100)
        tfidf.fit(all_about)

        emb_user = tfidf.transform(df['about_me_user'])
        emb_liked = tfidf.transform(df['about_me_liked'])
        about_me_similarity = [cosine_similarity(emb_user[i], emb_liked[i])[0, 0] for i in range(emb_user.shape[0])]
        df['about_me_similarity'] = about_me_similarity

        joblib.dump(tfidf, "tfidf.joblib")

        features = [
            'age_difference_flag',
            'same_city',
            'same_gender',
            'about_me_similarity',
            'matches_preferred_gender'
        ]
        X = df[features]
        y = df['target']

        print(f"📊 Training samples: {len(X)}")
        return X, y


    def train(self):
        """
        Train the recommender model and evaluate performance.

        Returns:
            tuple: (Trained XGBClassifier, fitted preprocessor)
        """
        X, y = self.create_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y
        )

        numerical_cols = [
            'age_difference_flag',
            'same_city',
            'same_gender',
            'about_me_similarity',
            'matches_preferred_gender'
        ]

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_cols)
        ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("📉 Training complete. Evaluation:")
        print(classification_report(y_test, y_pred))
        return model.named_steps['classifier'], model.named_steps['preprocessor']

