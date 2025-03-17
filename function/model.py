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
