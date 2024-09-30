import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  # To load a pre-trained scaler if available

import pandas as pd
import pickle

# Load model architecture

model_filename = './model/model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def main():
    st.title('Heart Disease Prediction')
    age = st.slider('Age', 18, 100, 50)
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)
    sex_num = 1 if sex == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs_options = ['False', 'True']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)

    with open('model/mean_std_values.pkl', 'rb') as f:
        mean_std_values = pickle.load(f)


    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
        
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()











# from tensorflow.keras.models import model_from_json
# with open('model\my_model_architecture.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)

# # Load model weights
# model.load_weights('model\my_model_weights.weights.h5')

# # Load pre-trained StandardScaler (if saved previously)
# # If you don't have a pre-trained scaler, ensure to train it using your dataset.
# # Assuming you saved the scaler using joblib.dump(scaler, 'scaler.pkl')
# try:
#     scal = joblib.load('scaler.pkl')
# except FileNotFoundError:
#     scal = StandardScaler()  # You will need to train it before use

# # Page Config
# st.set_page_config(page_title="Healthy Heart App", page_icon="⚕️", layout="centered", initial_sidebar_state="expanded")

# # Predicting the class
# def predict_disease(x): 
#     x_scaled = scal.transform([x])  # Scaling the input
#     return model.predict(x_scaled)

# # Preprocessing function
# def preprocess(age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):   
#     # Encode categorical variables
#     gender = 2 if gender == 'male' else 1
        
#     # Normalize categorical inputs for cholesterol and glucose
#     cholesterol_mapping = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
#     gluc_mapping = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    
#     cholesterol = cholesterol_mapping[cholesterol]
#     gluc = gluc_mapping[gluc]
    
#     # Create an array of inputs
#     x = np.array([age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active])

#     return x

# # HTML template for app header
# html_temp = """ 
#     <div style="background-color:pink;padding:13px"> 
#     <h1 style="color:black;text-align:center;">Healthy Heart App</h1> 
#     </div> 
#     """
# st.markdown(html_temp, unsafe_allow_html=True)
# st.subheader('by Mescoe Student')

# # User Inputs
# age_years = st.number_input("Age (in years)", min_value=1, max_value=100)
# age_days = age_years * 365  # Convert age in years to days

# gender = st.radio("Gender", ('male', 'female'))
# height = st.number_input("Height (cm)", min_value=50, max_value=250)
# weight = st.number_input("Weight (kg)", min_value=10, max_value=200)
# ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=300)
# ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=200)
# cholesterol = st.selectbox("Cholesterol Level", ("Normal", "Above Normal", "Well Above Normal"))
# gluc = st.selectbox("Glucose Level", ("Normal", "Above Normal", "Well Above Normal"))
# smoke = st.radio("Do you smoke?", (1, 0))
# alco = st.radio("Do you consume alcohol?", (1, 0))
# active = st.radio("Are you physically active?", (1, 0))

# # Preprocess the user input
# user_processed_input = preprocess(age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)

# # Predict heart disease risk
# if st.button("Predict"):
#     try:
#         pred = predict_disease(user_processed_input)
#         if pred[0] == 0:
#             st.success('You have a lower risk of heart disease!')
#         else:
#             st.error('Warning! You have a higher risk of heart disease!')
#     except Exception as e:
#         st.error(f"Error: {e}")

# # Sidebar Information
# st.sidebar.subheader("About App")
# st.sidebar.info("This web app helps you find out whether you are at risk of developing heart disease.")
# st.sidebar.info("Enter the required fields and click on the 'Predict' button to check your heart health.")
# st.sidebar.info("Don't forget to rate this app!")

# # App Feedback Slider
# feedback = st.sidebar.slider('How much would you rate this app?', min_value=0, max_value=5, step=1)
# if feedback:
#     st.header("Thank you for rating the app!")
#     st.info("Caution: This is just a prediction and not medical advice. Kindly see a doctor if you feel symptoms persist.")
