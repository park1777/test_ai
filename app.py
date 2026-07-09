import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 웹페이지 설정
st.set_page_config(page_title="당뇨병 예측 실험실", layout="wide")

# 사이드바: 과학 실험실 제어판
st.sidebar.header("🔬 과학 실험실 제어판")
st.sidebar.info("환자의 건강 데이터를 입력하여 당뇨병 확률을 분석합니다.")

# 입력값 슬라이더
pregnancies = st.sidebar.slider("임신 횟수", 0, 17, 3)
glucose = st.sidebar.slider("혈당 수치", 0, 200, 120)
blood_pressure = st.sidebar.slider("혈압", 0, 130, 70)
insulin = st.sidebar.slider("인슐린", 0, 900, 80)
bmi = st.sidebar.slider("BMI 지수", 0.0, 70.0, 25.0)

# 모델 생성 함수
def create_model():
    # 간단한 딥러닝 구조
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)), # 입력 피처 5개 기준
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # 이진 분류 (0~1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 메인 화면 디자인
st.title("🩺 피마 인디언 당뇨병 예측 시스템")
st.write("딥러닝 모델을 활용한 건강 상태 분석 도구입니다.")

# 예측 버튼
if st.button("예측 분석 시작"):
    # 입력 데이터 (사이드바 값 반영)
    input_data = np.array([[pregnancies, glucose, blood_pressure, insulin, bmi]])
    
    # 모델 실행
    model = create_model()
    prediction = model.predict(input_data)
    prob = prediction[0][0]

    # 결과 카드 표시
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="당뇨병 발병 확률", value=f"{prob*100:.2f}%")
    
    # 시각화 그래프
    fig, ax = plt.subplots()
    ax.bar(['정상', '당뇨병'], [1-prob, prob], color=['#3498db', '#e74c3c'])
    st.pyplot(fig)
    
    st.success("데이터 분석이 완료되었습니다!")

# 딥러닝 개념 설명
with st.expander("딥러닝 작동 원리 알아보기"):
    st.write("1. **ReLU**: 0 이하의 값은 차단하고 양수만 통과시켜 학습을 효율적으로 만듭니다.")
    st.write("2. **Sigmoid**: 결과를 0과 1 사이의 확률값으로 변환합니다.")
