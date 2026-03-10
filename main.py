# =============================================================================
# FRAUD DETECTION API
# =============================================================================
# This module exposes a REST API built with FastAPI that serves predictions
# from a pre-trained Random Forest model. Given a credit card transaction,
# it returns whether the transaction is fraudulent or not.
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# =============================================================================
# 1. APP INITIALIZATION & MODEL LOADING
# =============================================================================
app=FastAPI()
model=joblib.load("RandomForest_model.sav")
# =============================================================================
# 2. INPUT DATA MODEL
# =============================================================================
# Defines the expected structure of a transaction request.
# Features V1-V28 are anonymized PCA components from the original dataset.
class Credit_features(BaseModel):
    Time: float
    V1:float
    V2:float
    V3:float
    V4:float
    V5:float
    V6:float
    V7:float
    V8:float
    V9:float
    V10:float
    V11:float
    V12:float
    V13:float
    V14:float
    V15:float
    V16:float
    V17:float
    V18:float
    V19:float
    V20:float
    V21:float
    V22:float
    V23:float
    V24:float
    V25:float
    V26:float
    V27:float
    V28:float
    Amount:float

# =============================================================================
# 3. PREDICTION ENDPOINT
# =============================================================================

@app.post("/Credit_features")
def Post_tasks(Transaction_data:Credit_features):
    """
    Takes a credit card transaction as input and returns a fraud prediction.

    - **0** : Legitimate transaction
    - **1** : Fraudulent transaction
    """
    dictionnary= vars(Transaction_data)
    df=pd.DataFrame([dictionnary])
    Is_Fraud=model.predict(df)
    return {"prediction":int(Is_Fraud[0])}

