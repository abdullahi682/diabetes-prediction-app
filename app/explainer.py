import streamlit as st
import shap
import time
import matplotlib.pyplot as plt
from loader import model

def app(input_data):
    # Transform the input data using the feature engineering step.
    sample_transformed = model.named_steps['feature_engineering'].transform(input_data)
    
    # Create SHAP explainer for the final model.
    explainer = shap.TreeExplainer(model.named_steps['model'])
    
    # Compute SHAP values.
    shap_values = explainer.shap_values(sample_transformed)
    
    # Determine if SHAP returned a list (one per class) or a single array.
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values_for_sample = shap_values[1]
            base_value = explainer.expected_value[1]
        else:
            shap_values_for_sample = shap_values[0]
            base_value = explainer.expected_value
    else:
        shap_values_for_sample = shap_values
        base_value = explainer.expected_value
    
    # For a single sample, extract the first row.
    shap_values_sample = shap_values_for_sample[0]
    
    # Updated input streaming with the formatting from your screenshot.
    def stream_data():
        lines = [
            # Label in green italic, then value on a new line, plus a blank line
            f"<p style='color:green; font-style:italic;'>Pregnancies:</p>",
            f"<p>{float(input_data.iloc[0]['Pregnancies'])}</p><br>",

            f"<p style='color:green; font-style:italic;'>Glucose:</p>",
            f"<p>{float(input_data.iloc[0]['Glucose'])}</p><br>",

            f"<p style='color:green; font-style:italic;'>Insulin:</p>",
            f"<p>{float(input_data.iloc[0]['Insulin'])}</p><br>",

            f"<p style='color:green; font-style:italic;'>BMI:</p>",
            f"<p>{float(input_data.iloc[0]['BMI'])}</p><br>",

            f"<p style='color:green; font-style:italic;'>Age:</p>",
            f"<p>{float(input_data.iloc[0]['Age'])}</p><br>"
        ]
        # Stream each segment with a slight delay
        for line in lines:
            st.markdown(line, unsafe_allow_html=True)
            time.sleep(0.4)
    
    # Layout with two columns
    cols = st.columns(2)
    
    # Column 1: Stream user input with styled text
    with cols[0]:
        st.markdown("### Input Streaming")
        st.markdown("#### See your inputs in real-time below!")
        stream_data()
    
    # SHAP Waterfall Plot
    fig, ax = plt.subplots()
    explanation = shap.Explanation(
        values=shap_values_sample,
        base_values=base_value,
        data=sample_transformed.iloc[0],
        feature_names=sample_transformed.columns.tolist()
    )
    shap.plots.waterfall(explanation, show=False)
    fig.patch.set_facecolor("lightblue")
    fig.patch.set_alpha(0.3)
    ax.set_facecolor("#023047")
    ax.patch.set_alpha(0.5)
    
    # Column 2: Display SHAP Waterfall Plot
    with cols[1]:
        st.markdown("### SHAP Waterfall Plot")
        st.markdown(
            """
            - 🟡 **Base Value**: Expected model prediction without considering input features.
            - 🟡 **Feature Contributions**: Bars represent individual feature impact.
            - 🟡 **Output Prediction**: Sum of base value and contributions gives final output.
            """
        )
        st.pyplot(fig)
    
    # SHAP Force Plot
    if isinstance(shap_values, list):
        force_shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        force_base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value
    else:
        force_shap_values = shap_values
        force_base_value = explainer.expected_value
    
    force_plot_html = shap.force_plot(
        base_value=force_base_value,
        shap_values=force_shap_values[0],
        features=sample_transformed.iloc[0],
        feature_names=sample_transformed.columns.tolist()
    )
    
    st.markdown(
        """
        ### Column Explanations
        - 🟡 **Input Streaming**: Displays user inputs dynamically in real-time.
        - 🟡 **SHAP Waterfall Plot**: Visualizes how each feature contributes to the model prediction.
        - 🟡 **SHAP Force Plot**: Interactive plot showing positive/negative feature contributions.
        """,
        unsafe_allow_html=True,
    )
    
    force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
    st.markdown("### SHAP Force Plot")
    st.components.v1.html(force_plot_html, height=400)
