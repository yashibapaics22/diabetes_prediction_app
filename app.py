import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv(r"./diabetes.csv")

# Main title and description with updated lighter color theme
st.markdown("<h1 style='text-align: center; color: #76c7c0; font-family: Arial, sans-serif;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #777;'>This app predicts whether a patient is diabetic based on their health data.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 3px solid #76c7c0;'>", unsafe_allow_html=True)

# Sidebar header with light theme
st.sidebar.header('Enter Patient Data')
st.sidebar.write("<p style='color: #444; font-size: 16px;'>Please provide the following details for a diabetes checkup:</p>", unsafe_allow_html=True)

# Function for user input
def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

# User input
user_data = calc()

# Display patient data summary
st.subheader('Patient Data Summary:')
st.write(user_data)

# Model training and prediction
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Progress bar for model training
progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

# Prediction
result = rf.predict(user_data)

# Prediction result with light-themed styling
st.subheader('Prediction Result:')
if result[0] == 0:
    st.markdown("<div style='background-color:#A3D5D3;padding:20px;border-radius:10px;text-align:center;color:#333;font-size:20px;'>You are not Diabetic</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='background-color:#FFCCCB;padding:20px;border-radius:10px;text-align:center;color:#333;font-size:20px;'>You are Diabetic</div>", unsafe_allow_html=True)

# Display model accuracy with light theme
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader('Model Accuracy:')
st.markdown(f"<p style='text-align:center; font-size:20px; color:#333;'>Model Accuracy: <strong>{accuracy:.2f}%</strong></p>", unsafe_allow_html=True)
