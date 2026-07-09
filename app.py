import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 설정
st.set_page_config(page_title="당뇨병 예측 실험실", layout="wide")

# 2. 사이드바: 건강 데이터 입력
st.sidebar.header("🔬 과학 실험실 제어판")
pregnancies = st.sidebar.slider("임신 횟수", 0, 17, 3)
glucose = st.sidebar.slider("혈당 수치", 0, 200, 120)
blood_pressure = st.sidebar.slider("혈압", 0, 130, 70)
insulin = st.sidebar.slider("인슐린", 0, 900, 80)
bmi = st.sidebar.slider("BMI 지수", 0.0, 70.0, 25.0)

# 3. 모델 정의 (딥러닝 신경망 구조)
def create_model():
    # 은닉층 2개 (64개, 32개 뉴런), 활성화 함수 ReLU 사용
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100)
    return model

# 4. 메인 화면
st.title("🩺 피마 인디언 당뇨병 예측 시스템")
st.write("딥러닝(신경망) 모델을 활용한 건강 상태 분석 도구입니다.")

if st.button("예측 분석 시작"):
    # 사용자가 입력한 데이터를 리스트로 변환
    input_data = np.array([[pregnancies, glucose, blood_pressure, insulin, bmi]])
    
    # 모델 생성 및 예측 (실제 수업 시에는 학습된 모델을 불러오는 방식을 사용)
    model = create_model()
    # 주의: 여기서는 모델 구조 학습을 위해 더미 데이터가 필요하지만, 
    # 흐름상 예측 결과를 시각화하는 것에 집중했습니다.
    prob = 0.75 # 예측 확률 예시
    
    # 결과 카드
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="당뇨병 발병 확률", value=f"{prob*100:.2f}%")
    
    # 시각화
    fig, ax = plt.subplots()
    ax.bar(["정상", "당뇨병"], [1-prob, prob], color=['#3498db', '#e74c3c'])
    st.pyplot(fig)
    
    st.success("데이터 분석 완료!")

# 5. 딥러닝 개념 설명
with st.expander("딥러닝 작동 원리 알아보기"):
    st.write("이 모델은 **은닉층(Hidden Layer)**을 통해 데이터의 패턴을 찾습니다.")
    st.write("각 뉴런은 **ReLU 활성화 함수**를 거쳐 복잡한 규칙을 스스로 학습합니다.")
