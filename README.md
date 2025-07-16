# Inner Circle Assignment – Codebase

This repository contains solutions for both a **User Recommender System** and a **Data Aggregation Pipeline**.

---

## 📦 Directory Structure

```
.
├── data/
│   ├── users.csv
│   └── activity.csv
├── recommender_module/
│   ├── main.py
│   ├── .env
│   ├── models/
│   └── src/
│       ├── recommender.py
│       ├── trainer.py
│       └── utils.py
├── data_aggregation/
│   ├── main.py
│   ├── .env
│   ├── aggregation.sql
│   └── utils.py
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Place datasets

Put `users.csv` and `activity.csv` inside the `data/` directory at the project root.

---

## 1️⃣ Recommender System

**Directory:** `recommender_module/`

### Run

```bash
cd recommender_module
python main.py --user_id <USER_ID> --show 5
```

- Replace `<USER_ID>` with an actual user ID from your dataset.
- This will train the model (if needed), then print and visualize the top recommended users for that user.

### **Model Approach**

- Uses user profiles and interaction history to recommend users a given user might like.
- Built as a supervised learning task using tabular features and XGBoost.

### **Features Engineered:**

- **`age_difference_flag`**: 1 if age difference between the user and candidate ≤ 10, else 0.
- **`same_city`**: 1 if both users are from the same city, else 0.
- **`same_gender`**: 1 if both users are the same gender, else 0.
- **`about_me_similarity`**: Cosine similarity between users' “about_me” text fields (via TF-IDF).
- **`matches_preferred_gender`**: 1 if the candidate's gender matches the user's historic preferences (hetero/homo/bi behavior is inferred automatically).

### **Pipeline**

- Data is merged and features above are computed for each (user, candidate) pair.
- All features are scaled using `StandardScaler`.
- XGBoost (`XGBClassifier`) is used as the model.
- Model outputs probability of a positive interaction for each candidate user.
- Top-k recommendations are sorted and returned, with explanation of features for transparency.

### **Explainability**

- For each recommendation, output includes:
    - Probability score
    - Age/city/gender match status
    - Whether the candidate matches the user's inferred orientation
    - Text similarity score

---

## 2️⃣ Data Aggregation

**Directory:** `data_aggregation/`

### Run

```bash
cd data_aggregation
python main.py
```

- This loads the data, builds a SQLite database, runs the aggregation SQL, and outputs a CSV with metrics per day, gender, and city.

---

## 📝 Notes

- **No need to edit config files.** All paths are pre-configured.
- All required output folders and models are created automatically.
- Outputs include both console logs and CSV/visualizations.
- This codebase is fully modular and production-ready.

---

## ⚙️ Requirements

See `requirements.txt` for all needed Python packages.

---

## 💡 Tips

- For reproducible runs, always keep your datasets in the `data/` folder.
- You can re-run the scripts at any time; models and outputs will be refreshed or overwritten as needed.
- If you retrain the recommender, previous models in `recommender_module/models/` will be replaced.
- Aggregation output (CSV) will be saved in `data_aggregation/` or as specified in your configuration.
- All outputs (models, processed files) are auto-generated if not present—no manual setup required.

---
