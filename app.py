import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Diabetes Prediction Lab", layout="wide")

st.title("🩺 Pima Indian Diabetes Prediction")
st.write("Please enter the patient's health data below.")

# 1. st.form을 사용하여 입력창과 버튼을 하나로 묶음
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
    
    with col2:
        insulin = st.slider("Insulin", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    
    # 폼 안의 제출 버튼
    submit_button = st.form_submit_button(label="Start Prediction")

# 2. 버튼이 눌렸을 때만 로직 실행
if submit_button:
    input_data = np.array([[pregnancies, glucose, blood_pressure, insulin, bmi]])
    
    # 모델 처리 (시뮬레이션)
    model = MLPClassifier()
    prob = 0.75 
    
    st.divider() # 가로줄로 영역 구분
    st.subheader("Analysis Result")
    
    st.metric(label="Diabetes Risk Probability", value=f"{prob*100:.2f}%")
    
    fig, ax = plt.subplots()
    ax.bar(['Normal', 'Diabetes'], [1-prob, prob], color=['#3498db', '#e74c3c'])
    st.pyplot(fig)
    
    st.success("Analysis complete!")

with st.expander("Deep Learning Concepts"):
    st.write("The `st.form` keeps the UI clean until the user is ready to process the data.")
