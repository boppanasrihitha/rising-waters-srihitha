# Rising Waters — Flood Prediction System

A machine learning powered flood prediction system trained on historical
weather data (rainfall, cloud visibility, seasonal patterns) using Decision
Tree, Random Forest, KNN, and XGBoost, deployed via a Flask web application.

## Project Structure

```
flood_prediction/
├── training/
│   └── train_model.py       # Data prep + model training + comparison + save
├── app.py                    # Flask web application
├── requirements.txt
├── templates/
│   ├── home.html
│   ├── index.html
│   ├── chance.html
│   └── no_chance.html
└── static/
    ├── main.css
    └── main.js
```

After training, these files will be generated in the project root:
- `floods.save` — trained XGBoost model
- `transform.save` — fitted StandardScaler
- `feature_names.save` — list of feature column names used for training

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Step 1: Get the Dataset

Download the flood prediction dataset from Kaggle:
https://www.kaggle.com/arbethi/rainfall-dataset?select=flood+dataset.xlsx

Place the file inside the `training/` folder and rename it to
`flood_dataset.csv` (or `flood_dataset.xlsx` if you prefer Excel — just
update `DATA_PATH` at the top of `train_model.py`).

**Important:** Open `training/train_model.py` and update:
- `TARGET_COL = "class"` → set this to your dataset's actual target/label
  column name (e.g. `FLOODS`).

If no dataset file is found, the script will auto-generate a small synthetic
sample dataset so you can test the entire pipeline end-to-end immediately.

## Step 2: Train the Model

```bash
cd training
python train_model.py
```

This will:
1. Load and explore the dataset (`head()`, `info()`, `describe()`)
2. Handle missing values
3. Cap outliers using the IQR method
4. Encode any categorical columns
5. Split into train/test sets
6. Apply StandardScaler feature scaling
7. Train Decision Tree, Random Forest, KNN, and XGBoost
8. Print accuracy, confusion matrix, and classification report for each
9. Compare all models and select XGBoost as the final model
10. Save `floods.save`, `transform.save`, and `feature_names.save` to the
    project root

## Step 3: Run the Web Application

Go back to the project root (where `app.py` lives):

```bash
cd ..
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000/
```

Navigate to **Predict Floods**, enter the weather parameters (the form
fields are generated automatically based on your trained feature set), and
submit to view the flood risk result.

## Notes

- The `index.html` form automatically renders one input field per feature
  used during training (via `feature_names.save`), so no manual editing is
  required even if your dataset has a different number of features.
- For production deployment (e.g., IBM Cloud), set `debug=False` in
  `app.py` and use a production WSGI server such as `gunicorn`.
