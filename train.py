# =============================================================================
# FRAUD DETECTION MODEL - TRAINING PIPELINE
# =============================================================================
# This script trains a Random Forest Classifier to detect fraudulent credit
# card transactions. It applies undersampling to handle class imbalance,
# uses K-Fold cross-validation to evaluate model robustness, and performs
# hyperparameter tuning via RandomizedSearchCV.
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# 1. DATA LOADING
# =============================================================================

df = pd.read_csv("creditcard.csv")

# =============================================================================
# 2. CLASS BALANCING - UNDERSAMPLING
# =============================================================================
# The dataset is highly imbalanced (only 0.17% fraud cases).
# We apply undersampling by keeping all fraud cases and randomly sampling
# the same number of legitimate transactions.

df_fraud = df[df["Class"] == 1]
df_non_fraud = df[df["Class"] == 0]

df_fraud_sampled = df_fraud.sample(n=492, random_state=42)
df_non_fraud_sampled = df_non_fraud.sample(n=492, random_state=42)

df_balanced = pd.concat([df_fraud_sampled, df_non_fraud_sampled])

# =============================================================================
# 3. FEATURE / TARGET SPLIT
# =============================================================================

X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

# =============================================================================
# 4. K-FOLD CROSS-VALIDATION
# =============================================================================
# We use 5-fold cross-validation to evaluate model robustness before training
# the final model. We measure recall since detecting fraud (true positives)
# is more important than overall accuracy.

kf = KFold(n_splits=5, shuffle=True, random_state=42)
base_model = RandomForestClassifier()

cv_recall = cross_val_score(base_model, X, y, cv=kf, scoring='recall')

print("=" * 50)
print("CROSS-VALIDATION RESULTS (Recall)")
print("=" * 50)
for i, score in enumerate(cv_recall, 1):
    print(f"  Fold {i}: {score * 100:.2f}%")
print(f"  Mean Recall: {cv_recall.mean() * 100:.2f}%")
print("=" * 50)

# =============================================================================
# 5. TRAIN / TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================================
# 6. HYPERPARAMETER TUNING - RANDOMIZED SEARCH
# =============================================================================
# We search for the best hyperparameters using RandomizedSearchCV.

param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_distributions,
    cv=5,
    n_iter=10,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("\nBest hyperparameters found:")
print(random_search.best_params_)

# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n" + "=" * 50)
print("FINAL MODEL PERFORMANCE")
print("=" * 50)
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  Recall   : {recall * 100:.2f}%")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix - Fraud Detection")
plt.show()

# =============================================================================
# 8. MODEL SAVING
# =============================================================================

joblib.dump(best_model, "RandomForest_model.sav")
print("\nModel saved as 'RandomForest_model.sav'")