import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Alzheimer's Disease Prediction", layout="wide")

# Title and description
st.title("Alzheimer's Disease Prediction Tool")



#load XGBoost model
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as file:
        return pickle.load(file)


#load dataset to get mean values for missing features
@st.cache_data
def load_data():
    return pd.read_csv("alzheimers_disease_data.csv")


try:
    xgb_model = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    model_loaded = False

#features used during model training
model_features = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
    'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers',
    'CardiovascularDisease', 'Diabetes', 'Depression',
    'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]

#mapping of features for UI display
feature_mapping = {
    "Age": "Age",
    "Gender (0=Female, 1=Male)": "Gender",
    "Smoking Status (0=No, 1=Yes)": "Smoking",
    "Alcohol Consumption (0=No, 1=Yes)": "AlcoholConsumption",
    "Physical Activity (0=No, 1=Yes)": "PhysicalActivity",
    "Family History of Alzheimer's (0=No, 1=Yes)": "FamilyHistoryAlzheimers",
    "Cardiovascular Disease (0=No, 1=Yes)": "CardiovascularDisease",
    "Diabetes (0=No, 1=Yes)": "Diabetes",
    "Depression (0=No, 1=Yes)": "Depression",
    "Head Injury (0=No, 1=Yes)": "HeadInjury",
    "Hypertension (0=No, 1=Yes)": "Hypertension",
    "Memory Complaints (0=No, 1=Yes)": "MemoryComplaints",
    "Behavioral Problems (0=No, 1=Yes)": "BehavioralProblems",
    "Confusion (0=No, 1=Yes)": "Confusion",
    "Disorientation (0=No, 1=Yes)": "Disorientation",
    "Personality Changes (0=No, 1=Yes)": "PersonalityChanges",
    "Difficulty Completing Tasks (0=No, 1=Yes)": "DifficultyCompletingTasks",
    "Forgetfulness (0=No, 1=Yes)": "Forgetfulness",
    "BMI": "BMI",
    "Blood Pressure Level (0=Normal, 1=High)": "BloodPressureLevel",
    "Cholesterol Level (0=Normal, 1=High)": "CholesterolLevel",
    "Diet Quality (Rate from 1-10)": "DietQuality",
    "Sleep Quality (Rate from 1-10)": "SleepQuality",
    "Ease to perform Activities of Daily Living (ADL) (Rate from 1-10)": "ADL"
}

if model_loaded:
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)

    user_input_dict = {}

    with col1:
        st.markdown("### Demographics")
        user_input_dict["Age"] = st.number_input("Age", min_value=0, max_value=120, value=65)
        user_input_dict["Gender"] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        user_input_dict["BMI"] = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

    with col2:
        st.markdown("### Lifestyle Factors")
        user_input_dict["Smoking"] = st.selectbox("Smoking Status", [0, 1],
                                                  format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["AlcoholConsumption"] = st.selectbox("Alcohol Consumption", [0, 1],
                                                             format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["PhysicalActivity"] = st.selectbox("Physical Activity", [0, 1],
                                                           format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["DietQuality"] = st.slider("Diet Quality", min_value=1, max_value=10, value=5)
        user_input_dict["SleepQuality"] = st.slider("Sleep Quality", min_value=1, max_value=10, value=5)

    with col3:
        st.markdown("### Medical History")
        user_input_dict["FamilyHistoryAlzheimers"] = st.selectbox("Family History of Alzheimer's", [0, 1],
                                                                  format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["CardiovascularDisease"] = st.selectbox("Cardiovascular Disease", [0, 1],
                                                                format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["Diabetes"] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["Depression"] = st.selectbox("Depression", [0, 1],
                                                     format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["HeadInjury"] = st.selectbox("Head Injury", [0, 1],
                                                     format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["Hypertension"] = st.selectbox("Hypertension", [0, 1],
                                                       format_func=lambda x: "No" if x == 0 else "Yes")

    st.subheader("Symptoms & Health Indicators")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### Cognitive Symptoms")
        user_input_dict["MemoryComplaints"] = st.selectbox("Memory Complaints", [0, 1],
                                                           format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["Confusion"] = st.selectbox("Confusion", [0, 1],
                                                    format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["Disorientation"] = st.selectbox("Disorientation", [0, 1],
                                                         format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["Forgetfulness"] = st.selectbox("Forgetfulness", [0, 1],
                                                        format_func=lambda x: "No" if x == 0 else "Yes")
    with col5:
        st.markdown("### Behavioral Symptoms")
        user_input_dict["BehavioralProblems"] = st.selectbox("Behavioral Problems", [0, 1],
                                                             format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["PersonalityChanges"] = st.selectbox("Personality Changes", [0, 1],
                                                             format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["DifficultyCompletingTasks"] = st.selectbox("Difficulty Completing Tasks", [0, 1],
                                                                    format_func=lambda x: "No" if x == 0 else "Yes")
        user_input_dict["ADL"] = st.slider("Ease to perform Activities of Daily Living (ADL)", min_value=1,
                                           max_value=10, value=5)

    with col6:
        st.markdown("### Health Metrics")
        bp_level = st.selectbox("Blood Pressure Level", [0, 1], format_func=lambda x: "Normal" if x == 0 else "High")
        chol_level = st.selectbox("Cholesterol Level", [0, 1], format_func=lambda x: "Normal" if x == 0 else "High")

    if st.button("Predict Alzheimer's Risk"):
        with st.spinner("Processing..."):
            default_values = {}
            for feature in model_features:
                default_values[feature] = df[feature].mean()

            #handle blood pressure level
            if bp_level == 1:  # high
                default_values["SystolicBP"] = df["SystolicBP"].quantile(0.75)
                default_values["DiastolicBP"] = df["DiastolicBP"].quantile(0.75)
            else:  # normal bp
                default_values["SystolicBP"] = df["SystolicBP"].quantile(0.25)
                default_values["DiastolicBP"] = df["DiastolicBP"].quantile(0.25)

            #handle cholesterol level
            if chol_level == 1:  # high
                default_values["CholesterolTotal"] = df["CholesterolTotal"].quantile(0.75)
                default_values["CholesterolLDL"] = df["CholesterolLDL"].quantile(0.75)
                default_values["CholesterolTriglycerides"] = df["CholesterolTriglycerides"].quantile(0.75)
                default_values["CholesterolHDL"] = df["CholesterolHDL"].quantile(0.25)
            else:  # normal
                default_values["CholesterolTotal"] = df["CholesterolTotal"].quantile(0.25)
                default_values["CholesterolLDL"] = df["CholesterolLDL"].quantile(0.25)
                default_values["CholesterolTriglycerides"] = df["CholesterolTriglycerides"].quantile(0.25)
                default_values["CholesterolHDL"] = df["CholesterolHDL"].quantile(0.75)

            default_values.update(user_input_dict)

            input_df = pd.DataFrame([default_values])
            input_df = input_df[model_features]

            missing_cols = set(model_features) - set(input_df.columns)
            if missing_cols:
                st.warning(f"Missing columns: {missing_cols}")

            prediction = xgb_model.predict(input_df)

            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("Risk of Alzheimer's Disease Detected")
                st.write("The model indicates a higher risk of Alzheimer's disease based on the provided information.")
            else:
                st.success("No Risk of Alzheimer's Disease Detected")
                st.write("The model indicates a lower risk of Alzheimer's disease based on the provided information.")


with st.sidebar:
    st.title("About")
    st.info(
        "This application uses an XGBoost model trained on Alzheimer's disease data to predict risk "
        "based on various health and lifestyle factors."
    )
    st.subheader("Instructions")
    st.write("1. Fill in the patient information in all sections")
    st.write("2. Click 'Predict Alzheimer's Risk' to view results")
    st.write("3. Consult with healthcare professionals for proper diagnosis")