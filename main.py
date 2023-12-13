import streamlit as st
import pickle
import pandas as pd

# Load the pickled models
with open('model_dt.pkl', 'rb') as model_file:
    model_dt = pickle.load(model_file)

with open('model_knn.pkl', 'rb') as model_file:
    model_knn = pickle.load(model_file)

with open('model_svm.pkl', 'rb') as model_file:
    model_svm = pickle.load(model_file)

# Function to preprocess input and make predictions for each model
def predict_class_att_dt(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak]
    })

    prediction = model_dt.predict(input_data.values)  # Use .values to get the NumPy array from DataFrame
    return prediction[0]

def predict_class_att_knn(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak]
    })

    prediction = model_knn.predict(input_data.values)  # Use .values to get the NumPy array from DataFrame
    return prediction[0]

def predict_class_att_svm(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak]
    })

    prediction = model_svm.predict(input_data.values)  # Use .values to get the NumPy array from DataFrame
    return prediction[0]

# Streamlit app
st.title("Heart Disease Prediction App")

# Input features
age = st.slider("Age", 20, 80, 40)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.slider("Chest Pain Type (cp)", 0, 4, 1)
trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar (fbs)", [0, 1])
restecg = st.slider("Resting Electrocardiographic Results (restecg)", 0, 2, 1)
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 200, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest (oldpeak)", 0.0, 6.0, 1.0)

# Convert categorical input to numerical
sex = 1 if sex == 'Male' else 0
fbs = int(fbs)
exang = int(exang)

# Predict button for Decision Tree
if st.button("Predict (Decision Tree)"):
    result_dt = predict_class_att_dt(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak)
    st.success(f"The predicted class_att (Decision Tree) is: {result_dt}")

# Predict button for K-Nearest Neighbors
if st.button("Predict (K-Nearest Neighbors)"):
    result_knn = predict_class_att_knn(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak)
    st.success(f"The predicted class_att (K-Nearest Neighbors) is: {result_knn}")

# Predict button for Support Vector Machine
if st.button("Predict (Support Vector Machine)"):
    result_svm = predict_class_att_svm(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak)
    st.success(f"The predicted class_att (Support Vector Machine) is: {result_svm}")
