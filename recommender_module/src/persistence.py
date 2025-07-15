import joblib
import os

def save_model(model, preprocessor, model_path, preprocessor_path):
    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print("💾 Model and preprocessor saved.")

def load_model(model_path, preprocessor_path):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor
