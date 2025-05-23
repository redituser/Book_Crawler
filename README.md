
# 책 리뷰 감성 분석 시스템

## 1. 개요

### 1.1 목적
본 시스템은 네이버 책 검색 결과에서 수집한 리뷰 데이터를 LSTM 기반 딥러닝 모델을 활용하여 감성 분석하고, 그 결과를 시각적으로 표현하는 웹 애플리케이션입니다.

### 1.2 범위
- 네이버 쇼핑의 도서 리뷰 크롤링
- 수집된 리뷰 데이터의 감성 분석(긍정/중립/부정)
- 분석 결과의 시각화
### 1.3 시스템 구성도



### 1.3 대상 사용자
- 도서 트렌드 분석가
- 출판사 마케팅 담당자
- 독자 리뷰 분석에 관심 있는 일반 사용자

## 2. 시스템 아키텍처

### 2.1 주요 구성 요소
1. **데이터 수집 모듈** (Crawler.py)
   - Selenium 및 BeautifulSoup을 활용한 웹 크롤링
   - 도서 검색 및 리뷰 추출 기능

2. **감성 분석 모듈** (Mining.py)
   - LSTM 기반 딥러닝 모델 활용
   - 텍스트 전처리 및 분석 기능

3. **시각화 모듈** (Visualizer.py)
   - Matplotlib을 활용한 데이터 시각화
   - 리뷰 감성 분포 차트 생성

4. **사용자 인터페이스** (Main.py)
   - Streamlit 기반 웹 인터페이스
   - 검색 및 결과 표시 기능

5. **시스템 구성도**
   <br>
   ![image](https://github.com/user-attachments/assets/829bc57f-fa76-44b3-8202-d97d48cffaa4)


### 2.2 외부 라이브러리
- TensorFlow/Keras: 감성 분석 모델 구현
- Selenium/BeautifulSoup: 웹 크롤링
- Streamlit: 웹 인터페이스
- Matplotlib: 데이터 시각화



## 3. 데이터

### 3.1 입력 데이터
- 사용자가 입력한 검색 키워드

### 3.2 처리 데이터
- 웹 크롤링으로 수집한 도서 리뷰 텍스트
- 토큰화된 텍스트 시퀀스
- 패딩 처리된 숫자 배열

### 3.3 출력 데이터
- 감성 분석 결과 (감성 레이블, 확신도)
- 감성 분포 차트
- 리뷰 예시 목록

### 3.4 저장 데이터
- 사전 훈련된 감성 분석 모델 (sentiment_analysis_model.h5)
- 토크나이저 (tokenizer.pickle)
- 모델 설정 정보 (model_config.pickle)
