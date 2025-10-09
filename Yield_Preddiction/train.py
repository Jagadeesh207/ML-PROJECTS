import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

# === CONFIG ===
# Use Path; on Windows raw string recommended or use forward slashes
# Make sure your dataset is named 'synthetic_crop_dataset.csv' and is in the same folder
DATAFILE = Path("./synthetic_crop_dataset.csv")
OUTDIR = Path("./models")
OUTDIR.mkdir(parents=True, exist_ok=True)

# === 1. LOAD ===
if not DATAFILE.exists():
    raise FileNotFoundError(f"Dataset not found at {DATAFILE.resolve()}. Put your CSV there or change DATAFILE path.")

df = pd.read_csv(DATAFILE)
print("Loaded dataset shape:", df.shape)

# Drop 'ph' if present
if "ph" in df.columns:
    df = df.drop(columns=["ph"])

# Required features - ADJUSTED to match your dataset's column names
features = ["N", "P", "K", "temperature", "humidity", "rainfall", "Soil_Moisture"]
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Dataset missing required feature columns: {missing}")

# === 2. CLEAN & SPLIT ===
# ADJUSTED to use 'Yield' (capital Y)
df = df.dropna(subset=features + ["label", "Yield"]).reset_index(drop=True)
X = df[features].copy()
y_crop = df["label"].copy()
# ADJUSTED to use 'Yield' (capital Y)
y_yield = df["Yield"].astype(float).copy()

# Stratified split for classifier
X_train, X_test, y_crop_train, y_crop_test, y_yield_train, y_yield_test = train_test_split(
    X, y_crop, y_yield, test_size=0.2, random_state=RND, stratify=y_crop
)

# === 3. PREPROCESSING ===
# Label encode crop for classifier
le_crop = LabelEncoder()
y_crop_train_enc = le_crop.fit_transform(y_crop_train)
y_crop_test_enc = le_crop.transform(y_crop_test)
joblib.dump(le_crop, OUTDIR / "label_encoder.joblib")

# Numeric transformer (scaling)
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, features)])
preprocessor.fit(X_train)
joblib.dump(preprocessor, OUTDIR / "preprocessor.joblib")

X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# === 4. RANDOM FOREST CLASSIFIER ===
print("\nTraining RandomForestClassifier (crop recommendation)...")
rfc = RandomForestClassifier(random_state=RND, n_jobs=-1)

param_dist_clf = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_clf = RandomizedSearchCV(rfc, param_distributions=param_dist_clf, n_iter=8, cv=3,
                              scoring="accuracy", random_state=RND, n_jobs=-1, verbose=1)
grid_clf.fit(X_train_scaled, y_crop_train_enc)

best_rfc = grid_clf.best_estimator_
print("Best RF classifier params:", grid_clf.best_params_)

y_pred_clf = best_rfc.predict(X_test_scaled)
acc = accuracy_score(y_crop_test_enc, y_pred_clf)
print("RFC test accuracy:", acc)
print(classification_report(y_crop_test_enc, y_pred_clf, target_names=le_crop.classes_))
joblib.dump(best_rfc, OUTDIR / "rf_classifier.joblib")

# === 5. RANDOM FOREST REGRESSOR (Yield) ===
print("\nTraining RandomForestRegressor (yield)...")
# One-hot encode crop string labels for regressor inputs
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(y_crop_train.to_numpy().reshape(-1, 1))
joblib.dump(ohe, OUTDIR / "ohe_crop.joblib")

def build_regressor_matrix(X_df, crop_series):
    crop_ohe = ohe.transform(crop_series.to_numpy().reshape(-1, 1))
    X_scaled = preprocessor.transform(X_df)
    return np.hstack([X_scaled, crop_ohe])

X_reg_train = build_regressor_matrix(X_train, y_crop_train)
X_reg_test  = build_regressor_matrix(X_test, y_crop_test)

rfr = RandomForestRegressor(random_state=RND, n_jobs=-1)
param_dist_reg = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10]
}

# UPDATED: Using neg_mean_squared_error for better compatibility
grid_reg = RandomizedSearchCV(rfr, param_distributions=param_dist_reg, n_iter=8, cv=3,
                              scoring="neg_mean_squared_error", random_state=RND, n_jobs=-1, verbose=1)
grid_reg.fit(X_reg_train, y_yield_train)

best_rfr = grid_reg.best_estimator_
print("Best RF regressor params:", grid_reg.best_params_)

y_pred_reg = best_rfr.predict(X_reg_test)

# UPDATED: Manually calculate RMSE for compatibility
rmse_val = np.sqrt(mean_squared_error(y_yield_test, y_pred_reg))
print("RF regressor RMSE:", rmse_val)
print("RF regressor R2:", r2_score(y_yield_test, y_pred_reg))
joblib.dump(best_rfr, OUTDIR / "rf_regressor.joblib")

# === 6. ANN REGRESSOR (Keras) with anti-overfitting ===
print("\nTraining ANN regressor (Keras)...")
input_dim = X_reg_train.shape[1]

def build_ann_model(input_dim, lr=1e-3, dropout_rate=0.2, l2_reg=0.0):
    from tensorflow.keras import regularizers
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,), kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

ann = build_ann_model(input_dim, lr=1e-3, dropout_rate=0.25, l2_reg=1e-4)
es = EarlyStopping(monitor="val_root_mean_squared_error", patience=10, mode="min", restore_best_weights=True, verbose=1)
ckpt_path = OUTDIR / "ann_regressor_best.keras" # Using .keras extension
mc = ModelCheckpoint(str(ckpt_path), monitor="val_root_mean_squared_error", save_best_only=True, mode="min", verbose=1)

history = ann.fit(
    X_reg_train, y_yield_train,
    validation_split=0.15,
    epochs=200,
    batch_size=64,
    callbacks=[es, mc],
    verbose=2
)

ann_eval = ann.evaluate(X_reg_test, y_yield_test, verbose=0)
print("ANN test RMSE:", ann_eval[1])
ann.save(OUTDIR / "ann_regressor_final.keras") # Using .keras extension

# === 7. SUMMARY NOTES ===
notes = """
Anti-overfitting strategies applied:
- RandomForest: tune max_depth, min_samples_leaf, cross-validation.
- ANN: dropout, L2, BatchNormalization, EarlyStopping (restore_best_weights).
- Preprocessing: StandardScaler for numeric features, OneHotEncoder for crop.
Suggestions to further reduce overfitting:
- Use k-fold CV or repeated CV, feature selection, ensembling, per-crop regressors when appropriate.
"""
with open(OUTDIR / "training_notes.txt", "w") as fh:
    fh.write(notes)

print("\nAll models & preprocessors saved in:", OUTDIR.resolve())


