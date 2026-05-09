import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os

print("⚽ SportsPulse — Training Models...")

# ── LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("data/fifa23_clean.csv")
print(f"✓ {len(df)} players loaded")

# ── FEATURES ──────────────────────────────────────────────
FEATURES = [
    "overall", "potential", "age",
    "height_cm", "weight_kg",
    "pace", "shooting", "passing",
    "dribbling", "defending", "physic",
    "position_group", "league_level",
]

TARGET = "value_eur_m"

# Keep only rows with all features
df = df[FEATURES + [TARGET, "short_name"]].dropna()
print(f"✓ {len(df)} players after dropping nulls")

# Encode position group
le_pos = LabelEncoder()
df["position_group"] = le_pos.fit_transform(df["position_group"])

# ── SPLIT ─────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# ── MODEL 1 — LINEAR REGRESSION ───────────────────────────
print("\n📊 Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2  = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
print(f"   R²: {lr_r2:.4f} | MAE: €{lr_mae:.2f}M")

# ── MODEL 2 — RANDOM FOREST ───────────────────────────────
print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2  = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"   R²: {rf_r2:.4f} | MAE: €{rf_mae:.2f}M")

# ── MODEL 3 — XGBOOST ─────────────────────────────────────
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2  = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"   R²: {xgb_r2:.4f} | MAE: €{xgb_mae:.2f}M")

# ── RESULTS ───────────────────────────────────────────────
print("\n" + "="*45)
print("📊 MODEL COMPARISON")
print("="*45)
print(f"{'Model':<20} {'R²':>8} {'MAE':>12}")
print("-"*45)
print(f"{'Linear Regression':<20} {lr_r2:>8.4f} {lr_mae:>10.2f}M")
print(f"{'Random Forest':<20} {rf_r2:>8.4f} {rf_mae:>10.2f}M")
print(f"{'XGBoost':<20} {xgb_r2:>8.4f} {xgb_mae:>10.2f}M")
print("="*45)

# ── FEATURE IMPORTANCE ────────────────────────────────────
print("\n🎯 Top 10 Features (XGBoost):")
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)
for _, row in importance.head(10).iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"   {row['feature']:<20} {bar} {row['importance']:.4f}")

# ── SAVE MODELS ───────────────────────────────────────────
os.makedirs("models", exist_ok=True)
pickle.dump(xgb_model, open("models/xgb_model.pkl", "wb"))
pickle.dump(le_pos,    open("models/le_pos.pkl", "wb"))
pickle.dump(FEATURES,  open("models/features.pkl", "wb"))
print("\n✅ XGBoost model saved to models/")

# ── TEST ON FAMOUS PLAYERS ────────────────────────────────
print("\n🌟 Predictions on famous players:")
famous = df[df["short_name"].isin([
    "L. Messi", "K. Mbappé", "K. De Bruyne",
    "Neymar Jr", "R. Lewandowski"
])].head(10)

if len(famous) > 0:
    preds = xgb_model.predict(famous[FEATURES])
    for (_, row), pred in zip(famous.iterrows(), preds):
        actual = row["value_eur_m"]
        print(f"   {row['short_name']:<20} Actual: €{actual:.1f}M | Predicted: €{pred:.1f}M")