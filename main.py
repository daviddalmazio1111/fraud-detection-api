from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
app=FastAPI()
model=joblib.load("RandomForest_model.sav")
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



@app.post("/Credit_features")
def Post_tasks(Transaction_data:Credit_features):
    dictionnary= vars(Transaction_data)
    df=pd.DataFrame([dictionnary])
    Is_Fraud=model.predict(df)
    return {"prediction":int(Is_Fraud[0])}

