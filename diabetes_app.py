# diabetes_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and resources
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Try to load model info if available
        try:
            model_info = joblib.load('model_info.pkl')
        except:
            model_info = {'name': 'Optimized Model', 'performance': {}}
            
        return model, scaler, feature_names, model_info
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None

model, scaler, feature_names, model_info = load_model()

# App title and description
st.title("🩺 Diabetes Risk Assessment")
st.markdown("""
This tool assesses your risk of developing diabetes based on health metrics using advanced machine learning.
**Disclaimer:** This is a predictive tool, not a medical diagnosis. Always consult healthcare professionals.
""")

# Show warning if model files aren't loaded
if model is None:
    st.error("""
    ❌ Model files not found. Please make sure you have these files in the same directory:
    - diabetes_model.pkl
    - scaler.pkl
    - feature_names.pkl
    """)
    
    with st.expander("Setup Instructions"):
        st.markdown("""
        1. **Run the training script:**
           ```bash
           python model_training.py
           ```
        2. **This will create:**
           - diabetes_model.pkl (trained model)
           - scaler.pkl (feature scaler)
           - feature_names.pkl (feature names)
           - Various performance visualizations
        3. **Then run this app:**
           ```bash
           streamlit run diabetes_app.py
           ```
        """)
    st.stop()

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Health Assessment", "Model Information", "About Diabetes"])

with tab1:
    st.header("Health Information")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        pregnancies = st.slider("Number of Pregnancies", 0, 20, 0, 
                               help="Number of times pregnant")
        age = st.slider("Age (years)", 21, 100, 35, 
                       help="Age in years")
        
        st.subheader("Glucose Metabolism")
        glucose = st.slider("Glucose Level (mg/dL)", 50, 300, 100,
                           help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
        insulin = st.slider("Insulin Level (μU/mL)", 0, 300, 80,
                           help="2-Hour serum insulin level")
    
    with col2:
        st.subheader("Body Measurements")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 130, 70,
                                  help="Diastolic blood pressure")
        skin_thickness = st.slider("Skin Thickness (mm)", 5, 100, 25,
                                  help="Triceps skin fold thickness")
        bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 25.0, 0.1,
                       help="Body mass index (weight in kg/(height in m)^2)")
        
        st.subheader("Genetic Factor")
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01,
                       help="A function that scores the likelihood of diabetes based on family history")
    
    # Calculate derived features (same as in training)
    glucose_bmi = glucose * bmi / 1000
    age_glucose = age * glucose / 1000
    bp_bmi = blood_pressure * bmi / 1000
    insulin_glucose = insulin * glucose / 10000
    
    # Create input data in the correct order
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                 insulin, bmi, dpf, age,
                 glucose_bmi, age_glucose, bp_bmi, insulin_glucose]
    
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Prediction button
    if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
        try:
            # Preprocess the input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Calculate risk percentage
            risk_percentage = prediction_proba[0][1] * 100
            
            # Display results with appropriate styling
            st.subheader("Risk Assessment")
            
            if risk_percentage < 20:
                risk_level = "Very Low Risk"
                color = "green"
                emoji = "✅"
            elif risk_percentage < 40:
                risk_level = "Low Risk"
                color = "green"
                emoji = "✅"
            elif risk_percentage < 60:
                risk_level = "Moderate Risk"
                color = "orange"
                emoji = "⚠️"
            elif risk_percentage < 80:
                risk_level = "High Risk"
                color = "red"
                emoji = "❗"
            else:
                risk_level = "Very High Risk"
                color = "darkred"
                emoji = "🚨"
            
            st.markdown(f"<h2 style='color:{color};'>{emoji} {risk_level}</h2>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:{color};'>{risk_percentage:.1f}% probability</h3>", 
                       unsafe_allow_html=True)
            
            # Create a visual gauge
            st.progress(risk_percentage / 100)
            
            # Display detailed probability
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability of Diabetes", f"{risk_percentage:.1f}%")
            with col2:
                st.metric("Probability of No Diabetes", f"{100-risk_percentage:.1f}%")
            
            # Show key risk factors
            st.subheader("Key Risk Factors")
            
            risk_factors = []
            if glucose > 140:
                risk_factors.append(f"**High glucose level** ({glucose} mg/dL)")
            if bmi >= 30:
                risk_factors.append(f"**High BMI** ({bmi}) - Obese range")
            elif bmi >= 25:
                risk_factors.append(f"**Elevated BMI** ({bmi}) - Overweight range")
            if age > 45:
                risk_factors.append(f"**Age** ({age} years - increased risk category)")
            if dpf > 1.0:
                risk_factors.append(f"**Family history indication** (DPF: {dpf:.2f})")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No major risk factors identified based on your inputs.")
            
            # Recommendations based on risk level
            st.subheader("Recommendations")
            
            if risk_percentage < 40:
                st.success("""
                **Maintenance Plan:**
                - Continue with your healthy lifestyle habits
                - Maintain regular physical activity (150+ minutes/week)
                - Eat a balanced diet rich in fruits, vegetables, and whole grains
                - Schedule annual health check-ups
                """)
            elif risk_percentage < 70:
                st.warning("""
                **Prevention Plan:**
                - Consult with a healthcare provider for personalized advice
                - Increase physical activity to 30 minutes most days
                - Focus on weight management if needed
                - Reduce intake of processed foods and sugars
                - Consider getting a HbA1c test for baseline measurement
                """)
            else:
                st.error("""
                **Action Plan:**
                - **Schedule an appointment with a healthcare professional promptly**
                - Request comprehensive diabetes screening (fasting glucose, HbA1c)
                - Implement lifestyle changes immediately
                - Monitor your blood sugar levels regularly if advised
                - Join a diabetes prevention program if available
                """)
            
            # Medical disclaimer
            st.warning("""
            **Important Disclaimer:** This assessment is based on statistical models and should not replace 
            professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
            or other qualified health provider with any questions you may have regarding a medical condition.
            """)
            
        except Exception as e:
            st.error(f"An error occurred during assessment: {str(e)}")
            st.info("Please check that all input values are within the specified ranges.")

with tab2:
    st.header("Model Information")
    
    st.subheader("About the Prediction Model")
    st.write(f"**Model Type:** {model_info.get('name', 'Optimized Machine Learning Model')}")
    st.write(f"**Algorithm:** {type(model).__name__}")
    
    # Display model performance if available
    if 'performance' in model_info:
        st.subheader("Model Performance")
        perf = model_info['performance']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{perf.get('accuracy', 'N/A'):.4f}" if isinstance(perf.get('accuracy'), (int, float)) else "N/A")
        with col2:
            st.metric("ROC AUC", f"{perf.get('roc_auc', 'N/A'):.4f}" if isinstance(perf.get('roc_auc'), (int, float)) else "N/A")
        with col3:
            st.metric("F1 Score", f"{perf.get('f1', 'N/A'):.4f}" if isinstance(perf.get('f1'), (int, float)) else "N/A")
    
    st.subheader("Features Used")
    st.write("The model considers these health factors:")
    for i, feature in enumerate(feature_names, 1):
        st.write(f"{i}. {feature}")
    
    # Try to show feature importance if available
    try:
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df.head(8), use_container_width=True)
    except:
        pass

with tab3:
    st.header("About Diabetes")
    
    st.markdown("""
    ### What is Diabetes?
    
    Diabetes is a chronic condition that affects how your body turns food into energy. 
    There are two main types:
    
    - **Type 1 Diabetes**: An autoimmune condition where the body doesn't produce insulin
    - **Type 2 Diabetes**: A condition where the body doesn't use insulin properly (more common)
    - **Gestational Diabetes**: Develops during pregnancy and usually resolves after childbirth
    
    ### Key Risk Factors:
    
    - **High glucose levels** (≥126 mg/dL when fasting)
    - **Obesity or high BMI** (BMI ≥30 significantly increases risk)
    - **Family history** of diabetes (immediate relatives)
    - **High blood pressure** (≥140/90 mmHg)
    - **Age** (risk increases significantly after 45 years)
    - **Gestational diabetes** during pregnancy
    - **Physical inactivity**
    
    ### Normal Health Ranges:
    
    - **Glucose**: <100 mg/dL (fasting)
    - **BMI**: 18.5-24.9
    - **Blood Pressure**: <120/80 mmHg
    - **Insulin**: 2.6-24.9 μU/mL (fasting)
    
    ### Prevention Strategies:
    
    1. **Maintain a healthy weight**
    2. **Exercise regularly** (at least 150 minutes per week)
    3. **Eat a balanced diet** rich in fiber and low in processed foods
    4. **Limit sugar-sweetened beverages**
    5. **Avoid tobacco products**
    6. **Limit alcohol consumption**
    7. **Manage stress levels**
    8. **Get regular health check-ups**
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | **For educational purposes only**")

# Add a sidebar with additional information
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This diabetes risk assessment tool uses machine learning models trained on clinical data.
    
    **How it works:**
    1. Enter your health information
    2. The model analyzes multiple health factors
    3. Get your personalized risk assessment
    4. Receive evidence-based recommendations
    
    **Note:** This tool provides statistical assessment, not medical diagnosis.
    
    **Supported Algorithms:**
    - Random Forest
    - Logistic Regression
    - K-Nearest Neighbors
    - Support Vector Machine
    - Decision Tree
    """)
    
    # Display current model type
    st.subheader("Current Model")
    st.info(f"{type(model).__name__}")
    
    # Quick health tips
    st.subheader("Quick Health Tips")
    st.write("• Maintain healthy weight")
    st.write("• Exercise regularly")
    st.write("• Eat balanced diet")
    st.write("• Regular check-ups")