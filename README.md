# 🔍 Fraud Detection API

A machine learning application that detects fraudulent credit card transactions in real time. The project exposes a REST API built with FastAPI and a user-friendly interface built with Streamlit.

---

## 🌐 Live Demo

- **API** : https://fraud-detection-api-ayby.onrender.com/docs
- **Interface** : https://streamlit-interface-fraud-api.onrender.com/docs

---

## 🧠 How It Works

The model is trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. Since the original features are confidential, the dataset uses 28 anonymized PCA components (V1-V28) along with the transaction Time and Amount.

The dataset is highly imbalanced (only 0.17% fraud cases), so we applied **undersampling** to balance the classes before training a **Random Forest Classifier**.

**Model performance:**
- Accuracy: 94%
- Recall: 91%

---

## 🛠️ Tech Stack

- **Scikit-learn**: Model training and evaluation
- **FastAPI**: REST API to serve predictions
- **Streamlit**: User interface
- **Docker**: Containerization of the API
- **Render**: Cloud deployment

---

## 📁 Project Structure

- **train.py**: Loads the dataset, applies undersampling, trains the Random Forest model and saves it as a `.sav` file
- **main.py**: FastAPI application that loads the trained model and exposes a POST endpoint to make predictions
- **app.py**: Streamlit interface that collects transaction features from the user and displays the prediction
- **Dockerfile**: Defines the environment to run the FastAPI application in a container
- **requirements.txt**: Lists all Python dependencies

---

## 📊 Data Model

A transaction is composed of the following fields:

| Field | Type | Description |
|-------|------|-------------|
| Time | float | Seconds elapsed since the first transaction |
| V1-V28 | float | Anonymized PCA features |
| Amount | float | Transaction amount in dollars |
| Class | int | 0 = legitimate, 1 = fraud (target variable) |

---

## 🔌 API Routes

| Method | URL | Description |
|--------|-----|-------------|
| POST | /Credit_features | Takes transaction features and returns a fraud prediction |

### Predict a transaction
```
POST /Credit_features
```

Body:
```json
{
    "Time": 0.0,
    "V1": -1.35,
    "V2": -0.07,
    "V3": 2.53,
    "V4": 1.37,
    "V5": -0.33,
    "V6": 0.46,
    "V7": 0.23,
    "V8": 0.09,
    "V9": 0.36,
    "V10": 0.09,
    "V11": -0.55,
    "V12": -0.61,
    "V13": -0.99,
    "V14": -0.31,
    "V15": 1.46,
    "V16": -0.47,
    "V17": 0.20,
    "V18": 0.02,
    "V19": 0.40,
    "V20": 0.25,
    "V21": -0.01,
    "V22": 0.27,
    "V23": -0.11,
    "V24": 0.06,
    "V25": 0.12,
    "V26": -0.18,
    "V27": 0.13,
    "V28": -0.02,
    "Amount": 149.62
}
```

Response:
```json
{
    "prediction": 0
}
```

---

## ⚠️ Known Issues

- The PCA features (V1-V28) are anonymized, making the model difficult to interpret in a real business context
- The model was trained on undersampled data which may affect performance on real-world distributions

## 🔮 Future Improvements

- Implement SMOTE oversampling instead of undersampling
- Add more models and compare performance (XGBoost, LightGBM)
- Add feature importance visualization in the Streamlit interface
- Add user authentication to the API