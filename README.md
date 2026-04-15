# 🏡 Real Estate Investment Advisor

### Predicting Property Profitability & Future Value

---

## 📌 Project Overview

This project is a Machine Learning-based application designed to help users make smarter real estate investment decisions.
It predicts:

* ✅ Whether a property is a **Good Investment** (Classification)
* 💰 Estimated **Future Price after 5 Years** (Regression)

The system uses real estate data, performs preprocessing, trains ML models, and provides predictions through an interactive **Streamlit web app**.

---

## 🎯 Objectives

* Assist investors in evaluating property profitability
* Predict long-term property value growth
* Provide data-driven recommendations
* Build an end-to-end ML application

---

## 🧠 Technologies Used

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **XGBoost**
* **MLflow** (Experiment Tracking)
* **Streamlit** (Web Application)
* **Matplotlib, Seaborn** (Visualization)

---

## 📂 Project Structure

```
real_estate_advisor/
│
├── data/
│   └── india_housing_prices.csv
│
├── ml/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── eda.py
│
├── models/
│   ├── regressor.pkl
│   ├── classifier.pkl
│   ├── encoders.pkl
│   ├── features.pkl
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/real-estate-advisor.git
cd real-estate-advisor
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 📊 Step 1: Run EDA

```
cd ml
python eda.py
```

✔ Understand data distribution
✔ Analyze relationships between features

---

## 🤖 Step 2: Train Model

```
python train_model.py
```

✔ Trains XGBoost models
✔ Saves models in `/models`
✔ Logs experiments using MLflow

---

## 📈 Step 3: Run MLflow

```
mlflow ui
```

Open in browser:
👉 http://127.0.0.1:5000

✔ Track metrics (Accuracy, RMSE)
✔ Compare experiments

---

## 🌐 Step 4: Run Streamlit App

```
cd ../app
streamlit run app.py
```

✔ Enter property details
✔ Get investment prediction
✔ View future price estimation
✔ Explore feature importance & insights

---

## 📊 Features

* 🔍 Data preprocessing & cleaning
* 📈 Exploratory Data Analysis (EDA)
* 🤖 Machine Learning models:

  * XGBoost Classifier
  * XGBoost Regressor
* 📊 Visualizations & insights
* 🌐 Interactive Streamlit dashboard
* 📌 MLflow experiment tracking

---

## 📉 Model Evaluation

### Classification:

* Accuracy
* F1 Score

### Regression:

* RMSE
* R² Score

---

## ⚠️ Challenges & Solutions

| Challenge           | Solution                     |
| ------------------- | ---------------------------- |
| Data leakage        | Removed price-based features |
| High accuracy issue | Improved target logic        |
| Categorical data    | Applied Label Encoding       |
| Model tracking      | Integrated MLflow            |

---

## 🚀 Future Improvements

* Use real-time real estate APIs
* Add location-based prediction (maps)
* Improve model using hyperparameter tuning
* Deploy on cloud (Streamlit Cloud / AWS)

---

## 👨‍💻 Author

**Chaitanya**
Machine Learning Enthusiast

---

## ⭐ Acknowledgment

This project was developed as part of a capstone project to demonstrate end-to-end Machine Learning and deployment skills.

---

## 📌 Conclusion

This project successfully demonstrates how machine learning can be used to solve real-world problems in real estate investment by providing intelligent predictions and insights.

---
