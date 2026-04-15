import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier

from preprocessing import preprocess_data
from feature_engineering import create_targets

# Load data
df = pd.read_csv("../data/india_housing_prices.csv")

# Preprocess
df, encoders = preprocess_data(df)
df = create_targets(df)

# Features
X = df.drop([
    'ID',
    'Future_Price',
    'Good_Investment',
    'Price_in_Lakhs',        # 🔥 REMOVE
    'Price_per_SqFt'         # 🔥 REMOVE
], axis=1)
y_reg = df['Future_Price']
y_clf = df['Good_Investment']

# Split
X_train, X_test, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Models
reg_model = XGBRegressor(n_estimators=300, learning_rate=0.05)
clf_model = XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric='logloss')

# MLflow
mlflow.set_experiment("RealEstate_Project")

with mlflow.start_run():

    reg_model.fit(X_train, y_train_r)
    clf_model.fit(X_train, y_train_c)

    pred_r = reg_model.predict(X_test)
    pred_c = clf_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test_r, pred_r))
    acc = accuracy_score(y_test_c, pred_c)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("Accuracy", acc)

    mlflow.xgboost.log_model(reg_model, name = "regressor")
    mlflow.xgboost.log_model(clf_model, name = "classifier")

    print("RMSE:", rmse)
    print("Accuracy:", acc)

# Save models
joblib.dump(reg_model, "../models/regressor.pkl")
joblib.dump(clf_model, "../models/classifier.pkl")
joblib.dump(encoders, "../models/encoders.pkl")
joblib.dump(X.columns.tolist(), "../models/features.pkl")

print("✅ Models Saved!")