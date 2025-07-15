import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class UserRecommenderTrainer:
    def __init__(self, users_df, interactions_df):
        self.users_df = users_df
        self.interactions_df = interactions_df

    def create_training_data(self):
        print("📦 Creating training data...")
        df = self.interactions_df.merge(self.users_df, left_on='user_id', right_on='user_id')
        df = df.merge(self.users_df, left_on='liked_user_id', right_on='user_id', suffixes=('_user', '_liked'))

        df['age_difference'] = abs(df['age_user'] - df['age_liked'])
        df['same_city'] = (df['city_user'] == df['city_liked']).astype(int)
        df['same_gender'] = (df['gender_user'] == df['gender_liked']).astype(int)

        df['gender_user'] = df['gender_user'].astype(str)
        df['gender_liked'] = df['gender_liked'].astype(str)
        df['city_user'] = df['city_user'].astype(str)
        df['city_liked'] = df['city_liked'].astype(str)

        df['target'] = df['like_type'].apply(lambda x: 1 if x in [1, 2] else 0)

        # Upsample positives
        df_majority = df[df.target == 0]
        df_minority = df[df.target == 1]
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df = pd.concat([df_majority, df_minority_upsampled])

        features = [
            'age_user', 'age_liked', 'age_difference', 'same_city', 'same_gender',
            'gender_user', 'gender_liked', 'city_user', 'city_liked',
            'about_me_user', 'about_me_liked'
        ]

        X = df[features]
        y = df['target']

        # ✅ Fix NaNs in text columns
        X.loc[:, 'about_me_user'] = X['about_me_user'].fillna("")
        X.loc[:, 'about_me_liked'] = X['about_me_liked'].fillna("")

        print(f"📊 Training samples: {len(X)}")
        return X, y

    def train(self):
        X, y = self.create_training_data()
        print("🧪 Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

        categorical_cols = ['gender_user', 'gender_liked', 'city_user', 'city_liked']
        numerical_cols = ['age_user', 'age_liked', 'age_difference', 'same_city', 'same_gender']

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols),
            ('text_user', TfidfVectorizer(max_features=50), 'about_me_user'),
            ('text_liked', TfidfVectorizer(max_features=50), 'about_me_liked')
        ])

        print("🛠️ Training model...")
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss'))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("📉 Training complete. Evaluation:")
        print(classification_report(y_test, y_pred))
        return model.named_steps['classifier'], model.named_steps['preprocessor']
