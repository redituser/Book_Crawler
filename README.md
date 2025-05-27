
# 책 리뷰 감성 분석 시스템

## 1. 개요

### 1.1 목적
본 프로젝트는 네이버 책 검색 결과에서 수집한 리뷰 데이터를 LSTM 기반 딥러닝 모델을 활용하여 감성 분석하고, 그 결과를 시각적으로 표현하는 웹 애플리케이션입니다.

### 1.2 범위
- 네이버 쇼핑의 도서 리뷰 크롤링
- 수집된 리뷰 데이터의 감성 분석(긍정/중립/부정)
- 분석 결과의 시각화

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



## 3. 모델 학습 방법

### 3.1 KNU를 사용한 라벨링
![스크린샷 2025-05-27 233721](https://github.com/user-attachments/assets/a9bb2f5e-49b5-4604-99e5-9e98e29fc8a2)
-가장 오른쪽에 있는 숫자가 감성지수 이다 (0보다 크면 '좋음' 0 이면 '중립' -1 이하면 '나쁨')
### 라벨링 예시
      score_dict = { '좋아요':1 , '최고에요':1, '훌륭해요':1, '멋져요':1 , '별로예요':-1, '싫어요':-1, '나빠요':-1, '비싸요':-1 }
      str_review = '그 영화는 훌륭해요 멋져요 그래서 비싸요'
      def s_sentiment(sentence):
        sentence = sentence.split(' ')
        all_score = 0  
        for word,value in score_dict.items():
          for i in sentence:
            if i == word:
               all_score += value
        print(all_score)    
      
      s_sentiment(str_review)

### 3.2 데이터 전처리
      def map_sentiment(score):
       if score < 0: return 0    # 부정
       elif score == 0: return 1 # 중립  
       else: return 2           # 긍정

      #텍스트를 숫자 시퀀스로 변환
      tokenizer = Tokenizer()
      sequences = tokenizer.texts_to_sequences(reviews)
      padded_data = pad_sequences(sequences, maxlen=max_sequence_len)
### 3.3 모델 구조
      model = Sequential([
      Embedding(total_words, 64, input_length=max_sequence_len),
      Bidirectional(LSTM(64, return_sequences=True)),
      Bidirectional(LSTM(32)),
      Dense(32, activation='relu'),
      Dropout(0.5),
      Dense(3, activation='softmax')  # 3개 클래스 분류
])


### 3.4 학습 결과
![스크린샷 2025-05-20 125253](https://github.com/user-attachments/assets/258f72c4-6f7c-436d-951b-ac0f7d477869)

총 데이터: 304,027건<br>
테스트 정확도: 88.22%<br>
감성 분포: 긍정(134,894), 중립(126,675), 부정(42,458)



## 4. 핵심코드 

### 4.1 데이터 수집 (Crawler.py)
### 핵심 기능: 네이버 쇼핑에서 책 검색 후 리뷰 수집
<br>
     def find_book(text):
      
    # 1. 웹 드라이버로 네이버 쇼핑 접속
    driver = webdriver.Chrome()
    url = f"https://search.shopping.naver.com/book/search?query={text}"
    
    # 2. 검색 결과에서 책 링크들 추출
    elements = driver.find_elements(By.CSS_SELECTOR, 'a.bookListItem_info_top__DLxpl')
    hrefs = [el.get_attribute('href') for el in elements]
    
    # 3. 각 책 페이지 방문하여 리뷰 수집
    for detail_url in hrefs:
        # 리뷰 탭 클릭 → 더보기 버튼 클릭 → 리뷰 텍스트 추출
        reviews = soup.select('#book_section-review > ul > li > div.reviewItem_review__LEKrI > p')

### 4.2 감성 분석 (Mining.py)
### 핵심 기능: 수집된 리뷰를 모델로 감성 분석
<br>
     def analyze_sentiment(reviews):
    
    # 1. 사전 훈련된 모델과 토크나이저 불러오기
    model = tf.keras.models.load_model('sentiment_analysis_model.h5')
    tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
    
    # 2. 텍스트를 숫자로 변환 (토큰화)
    sequences = tokenizer.texts_to_sequences(reviews)
    
    # 3. 모든 리뷰 길이를 동일하게 맞춤 (패딩)
    padded_data = pad_sequences(sequences, maxlen=max_sequence_len)
    
    # 4. 모델로 감성 예측
    predictions = model.predict(padded_data)
    
    # 5. 결과를 이해하기 쉽게 변환
    sentiment_map = {0: "부정적", 1: "중립적", 2: "긍정적"}
    results = []
    for pred in predictions:
        sentiment_idx = np.argmax(pred)  # 가장 높은 확률의 감성
        sentiment = sentiment_map[sentiment_idx]
        confidence = float(pred[sentiment_idx])  # 확신도


### 4.3 웹 인터페이스 (Main.py)
### 핵심 기능: 사용자가 검색하고 결과를 확인할 수 있는 웹페이지
<br>
 
    1. 사용자로부터 검색어 입력받기
    keyword = st.sidebar.text_input("검색 키워드")

    if keyword:
    # 2. 크롤링으로 리뷰 수집
    reviews = nbc.find_book(keyword)
    
    # 3. 감성 분석 실행
    analysis = mining.analyze_sentiment(reviews)
    
    # 4. 결과를 차트와 표로 표시
    chart = viz.show_sentiment_pie(analysis['counts'])
    st.pyplot(chart)

## 5.결과

![스크린샷 2025-05-20 124221](https://github.com/user-attachments/assets/e961bda1-81ab-4be2-87ec-ed2fcce34c55)
![스크린샷 2025-05-20 124253](https://github.com/user-attachments/assets/160c71e0-8cfa-4677-bb0f-ece39e746435)
![스크린샷 2025-05-20 124329](https://github.com/user-attachments/assets/08dbe5cc-21e8-44a1-9c65-22e1c19387a1)
![스크린샷 2025-05-20 124418](https://github.com/user-attachments/assets/8000554a-c7ed-4ece-80ed-af877d37c538)
![스크린샷 2025-05-20 124357](https://github.com/user-attachments/assets/30d216bd-a9a2-47f7-a618-757237130669)

현재 결과는 다음과 같습니다

