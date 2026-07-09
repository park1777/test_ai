import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 웹페이지 설정
st.set_page_config(page_title="당뇨병 예측 실험실", layout="wide")

# 사이드바: 과학 실험실 제어판
st.sidebar.header("🔬 과학 실험실 제어판")
st.sidebar.info("환자의 건강 데이터를 입력하여 당뇨병 확률을 분석합니다.")

# 입력값 슬라이더 (실습용 예시)
pregnancies = st.sidebar.slider("임신 횟수", 0, 17, 3)
glucose = st.sidebar.slider("혈당 수치", 0, 200, 120)
blood_pressure = st.sidebar.slider("혈압", 0, 130, 70)
insulin = st.sidebar.slider("인슐린", 0, 900, 80)
bmi = st.sidebar.slider("BMI 지수", 0.0, 70.0, 25.0)

# 딥러닝 모델 정의 (ReLU와 Sigmoid 사용)
def create_model():
    model = tf.keras.models.Sequential([
        # 1. 입력층 및 첫 번째 은닉층: ReLU 활성화 함수 사용 (기울기 소실 방지)
        tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
        # 2. 두 번째 은닉층: ReLU 적용
        tf.keras.layers.Dense(32, activation='relu'),
        # 3. 출력층: Sigmoid 사용 (이진 분류를 위해 0~1 사이 확률값 출력)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # 학습 설정: Adam 최적화 및 Binary Cross-Entropy 손실 함수
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 메인 화면 디자인
st.title("🩺 피마 인디언 당뇨병 예측 시스템")
st.write("딥러닝 모델을 활용한 건강 상태 분석 도구입니다.")

# 예측 버튼
if st.button("예측 분석 시작"):
    # 가상의 입력 데이터 구성 (실제 사용 시 모델 학습 과정 필요)
    input_data = np.array([[pregnancies, glucose, blood_pressure, 0, 0, bmi, 0.2, 30]])
    
    # 모델 실행
    model = create_model()
    prediction = model.predict(input_data)
    prob = prediction[0][0]

    # 결과 카드 표시 (st.metric 활용)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="당뇨병 발병 확률", value=f"{prob*100:.2f}%")
    
    # 시각화 그래프
    fig, ax = plt.subplots()
    ax.bar(['정상', '당뇨병'], [1-prob, prob], color=['skyblue', 'salmon'])
    st.pyplot(fig)
    
    st.success("데이터 분석이 완료되었습니다!")

# 딥러닝 개념 설명
with st.expander("딥러닝 작동 원리 알아보기"):
    st.write("이 모델은 **ReLU 활성화 함수**를 통해 복잡한 비선형 관계를 학습합니다.")
    st.write("최종적으로 **Sigmoid 함수**를 통해 당뇨병일 확률을 0과 1 사이의 값으로 변환합니다.")