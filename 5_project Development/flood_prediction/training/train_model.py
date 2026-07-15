import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================================================
# 1. LOAD DATA
# ==================================================================
# Place your downloaded dataset (CSV or XLSX) in this same folder
# and update DATA_PATH accordingly.
DATA_PATH = "flood_dataset.csv"   # change to "flood_dataset.xlsx" if using Excel

if not os.path.exists(DATA_PATH):
    print(f"\n[WARNING] '{DATA_PATH}' not found. Generating a SAMPLE synthetic dataset")
    print("so you can test the full pipeline end-to-end. Replace this with your")
    print("real Kaggle flood dataset for actual predictions.\n")

    rng = np.random.default_rng(42)
    n = 500
    annual_rainfall = rng.normal(1200, 300, n).clip(200, 3000)
    cloud_visibility = rng.normal(5, 2, n).clip(0, 10)
    jun_sep_rainfall = rng.normal(800, 250, n).clip(100, 2500)
    temperature = rng.normal(28, 4, n).clip(10, 45)
    humidity = rng.normal(70, 15, n).clip(20, 100)

    # simple synthetic rule to create a target label
    risk_score = (
        (annual_rainfall > 1400).astype(int) +
        (jun_sep_rainfall > 1000).astype(int) +
        (humidity > 80).astype(int) +
        (cloud_visibility < 3).astype(int)
    )
    target = (risk_score >= 2).astype(int)

    dataset = pd.DataFrame({
        "ANNUAL_RAINFALL": annual_rainfall,
        "CLOUD_VISIBILITY": cloud_visibility,
        "JUN_SEP_RAINFALL": jun_sep_rainfall,
        "TEMPERATURE": temperature,
        "HUMIDITY": humidity,
        "class": target
    })
    dataset.to_csv(DATA_PATH, index=False)
    print(f"Sample dataset saved as '{DATA_PATH}'.\n")
else:
    if DATA_PATH.endswith(".xlsx"):
        dataset = pd.read_excel(DATA_PATH)
    else:
        dataset = pd.read_csv(DATA_PATH)

print("=== Data Preview ===")
print(dataset.head())
print("\n=== Data Info ===")
print(dataset.info())
print("\n=== Data Description ===")
print(dataset.describe())

# ==================================================================
# 2. HANDLE MISSING VALUES
# ==================================================================
print("\nMissing values before:\n", dataset.isnull().sum())
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
for col in dataset.select_dtypes(include='object').columns:
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
print("Missing values after:\n", dataset.isnull().sum())

# ==================================================================
# 3. HANDLE OUTLIERS (IQR CAPPING)
# ==================================================================
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper, upper, df[col])
    df[col] = np.where(df[col] < lower, lower, df[col])
    return df

TARGET_COL = "class"   # <-- change this to your actual target column name

numeric_cols = dataset.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != TARGET_COL:
        dataset = cap_outliers(dataset, col)

# ==================================================================
# 4. ENCODE CATEGORICAL COLUMNS
# ==================================================================
le = LabelEncoder()
for col in dataset.select_dtypes(include='object').columns:
    dataset[col] = le.fit_transform(dataset[col])

# ==================================================================
# 5. SPLIT X (independent) AND y (dependent/target)
# ==================================================================
X = dataset.drop(TARGET_COL, axis=1)
y = dataset[TARGET_COL]

FEATURE_NAMES = list(X.columns)
print("\nFeature columns used for training:", FEATURE_NAMES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================================================
# 6. FEATURE SCALING
# ==================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==================================================================
# 7. MODEL BUILDING FUNCTIONS
# ==================================================================
def decisiontree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- Decision Tree ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, y_pred


def randomForest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- Random Forest ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, y_pred


def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- KNN ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, y_pred


def xgboost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- XGBoost ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, y_pred


# ==================================================================
# 8. COMPARE MODELS
# ==================================================================
def compareModel(X_train, X_test, y_train, y_test):
    dt_model, dt_pred = decisiontree(X_train, X_test, y_train, y_test)
    rf_model, rf_pred = randomForest(X_train, X_test, y_train, y_test)
    knn_model, knn_pred = KNN(X_train, X_test, y_train, y_test)
    xgb_model, xgb_pred = xgboost_model(X_train, X_test, y_train, y_test)

    results = {
        "Decision Tree": accuracy_score(y_test, dt_pred),
        "Random Forest": accuracy_score(y_test, rf_pred),
        "KNN": accuracy_score(y_test, knn_pred),
        "XGBoost": accuracy_score(y_test, xgb_pred),
    }

    print("\n=== Model Comparison ===")
    for k, v in results.items():
        print(f"{k}: {v*100:.2f}%")

    return xgb_model  # XGBoost selected as final deployment model


final_model = compareModel(X_train, X_test, y_train, y_test)

# ==================================================================
# 9. SAVE MODEL, SCALER, AND FEATURE NAMES
# ==================================================================
joblib.dump(final_model, "../floods.save")
joblib.dump(sc, "../transform.save")
joblib.dump(FEATURE_NAMES, "../feature_names.save")

print("\nModel, Scaler, and Feature Names saved successfully:")
print(" - floods.save")
print(" - transform.save")
print(" - feature_names.save")
