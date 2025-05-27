import pandas as pd
import joblib
from PIL import Image
from data.config import thresholds
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load the data
data = pd.read_csv('datasets/diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09,
                                                    random_state=42, stratify=y)
# Create a StratifiedKFold object for cross-validation (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load additional resources
page_icon = Image.open(r"C:\Users\X1 CARBON TOUCH\Desktop\project-experiementation\Diabetes-Prediction\image\page_icon.jpeg")
model = joblib.load(r'C:\Users\X1 CARBON TOUCH\Desktop\project-experiementation\Diabetes-Prediction\diabetes_prediction_pipeline.joblib')

# Evaluate the loaded model using the test set
y_score = model.predict_proba(X_test)[:, 1]
y_pred = (y_score >= thresholds).astype(int)

# Compute performance metrics on the test set
accuracy_result = round(accuracy_score(y_test, y_pred) * 100, 2)
f1_result = (f1_score(y_test, y_pred) * 100).round(2)
recall_result = (recall_score(y_test, y_pred) * 100).round(2)
precision_result = (precision_score(y_test, y_pred) * 100).round(2)
roc_auc = (roc_auc_score(y_test, y_score) * 100).round(2)
