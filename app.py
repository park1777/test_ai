import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Diabetes Prediction Lab", layout="wide")

st.title("🩺 Pima Indian Diabetes Prediction")

# 1. 세션 상태 초기화 (처음 실행 시 확률을 저장할 공간 만들기)
if 'prob' not in st.session_state:
    st.session_state.prob = 0.5  # 초기값

# 2. 폼 정의
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
    with col2:
        insulin = st.slider("Insulin", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    
    submit_button = st.form_submit_button(label="Start Prediction")

# 3. 버튼을 눌렀을 때만 로직 실행 및 결과 업데이트
if submit_button:
    # (여기서 실제 모델 예측 로직 수행)
    # 예시: 입력값에 따른 가상의 확률 계산 (실제 모델 학습 시 예측값으로 대체)
    new_prob = (glucose / 300) + (bmi / 200)
    st.session_state.prob = min(new_prob, 1.0) # 1을 넘지 않게 설정

# 4. 결과 출력 (세션 상태에 저장된 값을 가져와서 표시)
st.metric(label="Diabetes Risk Probability", value=f"{st.session_state.prob*100:.2f}%")

fig, ax = plt.subplots()
ax.bar(['Normal', 'Diabetes'], [1-st.session_state.prob, st.session_state.prob], color=['#3498db', '#e74c3c'])
st.pyplot(fig)
