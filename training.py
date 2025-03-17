import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from function.model import Model  # Assuming this Model is defined elsewhere

# Load the data
data = pd.read_csv(r'C:\Users\X1 CARBON TOUCH\Desktop\project-experiementation\Diabetes-Prediction\datasets\diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09,
                                                    random_state=42, stratify=y)
# Create a StratifiedKFold object for cross-validation (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train the model using the training split
Model.fit(X_train, y_train)

# Evaluate the model on the test split
y_pred = Model.predict_proba(X_test)[:, 1]
print("ROC_AUC Score: ", (roc_auc_score(y_test, y_pred) * 100).round(2))

# Save the trained model
joblib.dump(Model, r'C:\Users\X1 CARBON TOUCH\Desktop\project-experiementation\Diabetes-Prediction\diabetes_prediction_pipeline.joblib')

# (Optional) Pipeline configuration using GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from function.transformers import FeatureEngineering, WoEEncoding, ColumnSelector

selected_columns = [
    'Pregnancies', 'Glucose', 'BMI', 'PregnancyRatio',
    'RiskScore', 'InsulinEfficiency', 'Glucose_BMI', 'BMI_Age',
    'Glucose_woe', 'RiskScore_woe'
]

# Pipeline setup using GradientBoostingClassifier with a valid criterion.
Model = Pipeline([
    ('feature_engineering', FeatureEngineering()),
    ('woe_encoding', WoEEncoding()),
    ('column_selector', ColumnSelector(selected_columns)),
    ('model', GradientBoostingClassifier(
         max_depth=6,
         n_estimators=300,
         criterion='friedman_mse',  # Valid for GradientBoostingClassifier.
         random_state=42))
])
