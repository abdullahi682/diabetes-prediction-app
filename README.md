# Diabetes Prediction with AI


![app.gif](image/app.gif)


This project demonstrates a machine learning solution for predicting diabetes based on user-provided health data. The application uses **Streamlit** for an interactive web interface and advanced interpretability tools like SHAP and permutation importance to explain model predictions.

## Live Demo

Check out the live application: [Diabetes Prediction App](https://diabetes-prediction-app-m6qd.onrender.com/)

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Features](#features)
5. [Installation](#installation)
6. [How It Works](#how-it-works)
7. [Project Structure](#project-structure)
8. [Explanation Methods](#explanation-methods)
9. [Model Performance](#model-performance)
10. [Project Motivation](#project-motivation)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contacts](#contacts)

---

## Overview

The **Diabetes Prediction with AI** project leverages a machine learning model to predict diabetes risk. Built with **Streamlit**, the app explains predictions using SHAP and permutation importance while showcasing model performance metrics. This model has not been reviewed by medical professionals; it is developed solely for experimental and testing purposes.
The model was developed based on the ROC AUC metric, while efforts were made to improve the Recall metric when selecting the threshold, as this decision was made due to the medical context.

### Why This Project?

Understanding diabetes risk through data-driven predictions can help identify potential cases early. This project also demonstrates:
- Practical application of machine learning.
- Model interpretability through SHAP and permutation importance.
- Real-world deployment of machine learning models.
-Practical Deployment: The project demonstrates end-to-end deployment—from data preprocessing to live prediction via a web interface.
-Early Warning: Data-driven predictions can help in early identification of potential diabetes cases.

---

## Dataset

The dataset is sourced from the **National Institute of Diabetes and Digestive and Kidney Diseases**. It includes:

The dataset contains the following details:

### General Overview
- **Number of rows:** 768
- **Number of columns:** 9
- **Column names and data types:**
  - `Pregnancies` (int64): Number of times pregnant.
  - `Glucose` (int64):  Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
  - `BloodPressure` (int64): Diastolic blood pressure (mm Hg).
  - `SkinThickness` (int64): Triceps skin fold thickness (mm).
  - `Insulin` (int64): 2-Hour serum insulin (mu U/ml).
  - `BMI` (float64): Body mass index (weight in kg/(height in m)^2).
  - `DiabetesPedigreeFunction` (float64): Diabetes pedigree function.
  - `Age` (int64): Age (years).
  - `Outcome` (int64): Class variable (0 or 1).

### Sample Data (First 5 Rows)
| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin |  BMI  | DiabetesPedigreeFunction | Age | Outcome |
|-------------|---------|---------------|---------------|---------|-------|---------------------------|-----|---------|
| 6           | 148     | 72            | 35            | 0       | 33.6  | 0.627                     | 50  | 1       |
| 1           | 85      | 66            | 29            | 0       | 26.6  | 0.351                     | 31  | 0       |
| 8           | 183     | 64            | 0             | 0       | 23.3  | 0.672                     | 32  | 1       |
| 1           | 89      | 66            | 23            | 94      | 28.1  | 0.167                     | 21  | 0       |
| 0           | 137     | 40            | 35            | 168     | 43.1  | 2.288                     | 33  | 1       |

### Statistical Summary
- **Pregnancies:** Mean = 3.85, Max = 17
- **Glucose:** Mean = 120.89, Min = 0 (possible missing values)
- **BloodPressure:** Mean = 69.11, Min = 0 (possible missing values)
- **SkinThickness:** Mean = 20.54, Min = 0 (possible missing values)
- **Insulin:** Mean = 79.80, Min = 0 (possible missing values)
- **BMI:** Mean = 31.99, Min = 0 (possible missing values)
- **DiabetesPedigreeFunction:** Mean = 0.47, Max = 2.42
- **Age:** Mean = 33.24, Max = 81
- **Outcome:** Proportion of `1` (positive diabetes) = 34.9%


#### We use only `Pregnancies`, `Glucose`, `BMI`, `Insulin`, `Age` for prediction.
---

## Model
You can learn more about the model in detail from [here](notebooks/Model.ipynb). The `GradientBoostingClassifier` model was chosen through experimentation and showed the best performance.
1. Stability & Generalization
Overfitting Control: Unlike Random Forest, which may sometimes be prone to overfitting, Gradient Boosting builds trees sequentially and optimizes for errors made by previous models. This helps in better generalization, which is crucial when dealing with real-world, unseen data.
Robust Performance on Noisy Data: Since Gradient Boosting focuses on correcting errors iteratively, it is often more stable than XGBoost when dealing with noise in data.
2. Interpretability & Feature Importance
Better Feature Attribution: Gradient Boosting is known for generating feature importance that can be easily interpreted using SHAP (Shapley Additive Explanations), as seen in your explainer.py file​explainer. This allows domain experts and healthcare professionals to understand what factors contribute most to the predictions.
3. Performance Beyond Accuracy (ROC AUC)
Strong ROC AUC Score (95.37%): Even though its accuracy is slightly lower than XGBoost and Random Forest, Gradient Boosting has the highest ROC AUC score (95.37%), meaning it is better at distinguishing between positive and negative cases. This is especially crucial in medical applications like diabetes prediction, where precision in identifying high-risk patients is more important than just accuracy.
4. Computational Efficiency
Less Memory Intensive than Random Forest: Gradient Boosting typically requires fewer trees than Random Forest to achieve comparable performance, making it a better choice for deployment in resource-constrained environments.
Faster Training than XGBoost: While XGBoost is an optimized implementation, its hyperparameter tuning and tree-pruning mechanisms can be computationally expensive.
5. Better Handling of Class Imbalances
In real-world applications like diabetes prediction, datasets often contain imbalanced classes (more non-diabetic than diabetic cases). Gradient Boosting handles such imbalances better due to its iterative re-weighting mechanism.
The required hyperparameters were identified using the `optuna` optimizer. For the model to function, it needs `FeatureEngineering`, `WoEEncoding`, and `ColumnSelector` transformers, which are combined through a pipeline.
`Cross-validation` and `ROC AUC` were used for model selection because the number of observations was small, and splitting into test/train sets would have been inaccurate.
The final prediction pipeline is built using Gradient Boosting and incorporates several custom transformers to enhance feature quality:

FeatureEngineering: Creates new features (e.g., PregnancyRatio, RiskScore, InsulinEfficiency) that capture underlying relationships in the data.
WoEEncoding: Transforms selected features into their Weight of Evidence (WoE) representation, improving interpretability.
ColumnSelector: Selects the most relevant engineered features for the final model.
The pipeline was constructed after experimenting with multiple models (including SVM, Decision Tree, and Random Forest) and was ultimately evaluated using cross-validation with the ROC AUC metric. The final model is saved as diabetes_prediction_pipeline.joblib for deployment.

### About tarnsformers
#### **1. FeatureEngineering**
Transforms raw data into a format suitable for machine learning. This includes scaling, encoding, creating new features, or handling missing data.


#### **2. WoEEncoding (Weight of Evidence Encoding)**
Features must help to better explain the `Outcome` after WoE.
The Weight of Evidence (WoE) for a category in a feature is calculated as:

Where:
- `P(Feature = X | Target = 1)`: Proportion of positive cases (`Target = 1`) for the category `X`.
- `P(Feature = X | Target = 0)`: Proportion of negative cases (`Target = 0`) for the category `X`.

##### Example:
If a feature `X` has the following counts:
- For `Target = 1` (Positive): `N1`
- For `Target = 0` (Negative): `N0`

#### **3. ColumnSelector**
Selects specific columns *Pregnancies*, *Glucose*, *BMI*, *PregnancyRatio*,
    *RiskScore*, *InsulinEfficiency*, *Glucose_BMI*, *BMI_Age*,
    *Glucose_woe*, *RiskScore_woe* after `FeatureEngineering`, it helps remove noice columns.

---
## Features

1. **Interactive Input**: Enter health parameters (Pregnancies, Glucose, Insulin, BMI, Age).
2. **Diabetes Prediction**: Real-time risk prediction with probability.
3. **SHAP Explanations**: Visualize individual prediction explanations using:
   - Waterfall Plot
   - Force Plot

4.**Permutation Importance**: Explore which features have the highest impact on the model’s decisions.
**Performance Metrics**: View accuracy, precision, recall, F1 score, and ROC AUC via interactive charts.
**Educational Content**: An "About" section provides insights on diabetes risk factors and model methodology.
5. **Permutation Importance**: Analyze which features most influence the predictions.
6. **Performance Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
7.**Real-Time Prediction**: The trained pipeline predicts diabetes risk on the fly.
6. **Informational Section**: Learn about diabetes risk factors in the "About" section.

---

## Installation

### Prerequisites
- Python 3.10 or above
- Pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/abdullahi682/diabetes-prediction-app/tree/main
   cd Diabetes-Prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```bash
   streamlit run main.py
   ```

---

## How It Works

### Application Workflow
1. **User Input**:
   - Enter health data in the sidebar.
   - Features: Pregnancies, Glucose, Insulin, BMI, Age.
2. **Prediction**:
   - The trained model predicts diabetes risk and displays the result.
3. **Explanation**:
   - View SHAP plots (Waterfall and Force) for detailed feature contributions.
   - Explore permutation importance for global feature analysis.
4. **Model Performance**:
   - Metrics such as Accuracy, F1 Score, and ROC AUC are displayed.


# Project Structure
```
Diabetes-Prediction/
├── README.md                 # Project documentation
├── main.py                   # Entry point for the Streamlit app
├── loader.py                 # Data loading and preprocessing
├── training.py               # Script for training the model
├── requirements.txt          # Project dependencies
├── LICENSE                   # License file
├── datasets/
│   ├── diabetes.csv          # Dataset used for training and predictions
├── models/
│   ├── diabetes_prediction_pipeline.joblib             # Trained machine learning model
├── images/
│   ├── page_icon.jpeg        # Application page icon
├── data/
│   ├── config.py             # Configuration variables
│   ├── base.py               # Static HTML/CSS content
├── function/
│   ├── model.py              # Custom model implementation
│   ├── function.py           # Utility functions
└── app/                      # Application logic and components
    ├── predict.py            # Prediction logic
    ├── explainer.py          # SHAP-based explanations
    ├── perm_importance.py    # Permutation importance analysis
    ├── performance.py        # Visualization of model performance metrics
    ├── input.py              # User input handling for predictions
    ├── about.py              # Informational section on diabetes
```


---

## Explanation Methods

1. **SHAP Waterfall Plot**:
   - Shows how each feature contributes positively or negatively to the prediction.
2. **SHAP Force Plot**:
   - Interactive visualization of feature contributions to individual predictions.
3. **Permutation Importance**:
   - Ranks features by their impact on the model's predictions.

---

## Model Performance

Performance metrics calculated:
- **Accuracy**: Percentage of correct predictions. (0.8571)
- **Precision**: Ratio of true positives to total positive predictions. (0.7692)
- **Recall**: Ratio of true positives to total actual positives. (0.8333)
- **F1 Score**: Harmonic mean of Precision and Recall. (0.8000)
- **ROC AUC**: Area under the ROC curve. (0.8904)

Metrics are displayed as donut charts in the application.

---

## Project Motivation

This project was developed to:
- Build knowledge in machine learning, especially in healthcare.
- Gain hands-on experience with model interpretability techniques like SHAP.
- Deploy an AI solution using **Streamlit**.

---

## Contributing

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push:
   ```bash
   git commit -m "Feature description"
   git push origin feature-name
   ```
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contacts

If you have any questions or suggestions, please contact:
- Email: updulze29@gmail.com
- GitHub Issues: [Issues section](https://github.com/AbdullahiAhm/Diabetes-Prediction/issues)
- GitHub Profile: [abdullahi682](https://github.com/abdullahi682/)
- Linkedin: [Abdullahi Ahmed](https://www.linkedin.com/in/AbdullahiAhm/)


### <i>Thank you for your interest in the project!</i>
