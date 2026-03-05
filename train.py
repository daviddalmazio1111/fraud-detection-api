import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

df=pd.read_csv("creditcard.csv")
#Split the dataset in two
df_Fraud=df[df["Class"]==1]
df_Non_Fraud=df[df["Class"]==0]
#Filtered the two dataset
df_Fraud_Filtered=df_Fraud.sample(n=492)
df_Non_Fraud_Filtered=df_Non_Fraud.sample(n=492)
#Merge the two filtered dataset
df_undersampled=pd.concat([df_Fraud_Filtered,df_Non_Fraud_Filtered])
#Creation of the train and test set
X=df_undersampled.drop("Class",axis=1)
y=df_undersampled["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
#Model training
"""
Model=RandomForestClassifier()
Model.fit(X_train,y_train)
#Evaluation of the model
y_pred=Model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
print("Accuracy:",accuracy)
print("Recall:",recall)
"""
print(type(X_test))
#save model
#filename="RandomForest_model.sav"
#joblib.dump(Model,filename)
