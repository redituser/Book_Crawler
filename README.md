# 책 리뷰 감성 분석 시스템

## 1. 개요

### 1.1 목적
본 프로젝트는 네이버 책 검색 결과에서 수집한 리뷰 데이터를 **BERT 기반의 딥러닝 모델**을 활용하여 감성 분석하고, 그 결과를 시각적으로 표현하는 웹 애플리케이션입니다.

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
    - **`klue/bert-base` 모델 기반의 Fine-tuning**
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
- **Transformers**: BERT 모델 로딩 및 토큰화
- Selenium/BeautifulSoup: 웹 크롤링
- Streamlit: 웹 인터페이스
- Matplotlib: 데이터 시각화



## 3. 모델 학습 방법

~~### 3.1 KNU를 사용한 라벨링~~
~~![스크린샷 2025-05-27 233721](https://github.com/user-attachments/assets/a9bb2f5e-49b5-4604-99e5-9e98e29fc8a2)~~
~~-가장 오른쪽에 있는 숫자가 감성지수 입니다~~

~~### 라벨링 예시~~
```python
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
```
### 라벨링(개선)
## LM 스튜디오의 GEMMA 를 사용하여 긍정,중립,부정 의 결과값을 csv파일로 저장
![image](https://github.com/user-attachments/assets/6bf7a376-1e8d-4516-8b96-11638ba2568a)



### 3.2 데이터 전처리
```python
def map_sentiment(score):
    if score < 0: return 0   # 부정
    elif score == 0: return 1 # 중립
    else: return 2           # 긍정

# (기존) 텍스트를 숫자 시퀀스로 변환
# tokenizer = Tokenizer()
# sequences = tokenizer.texts_to_sequences(reviews)
# padded_data = pad_sequences(sequences, maxlen=max_sequence_len)

# (개선) BERT 토크나이저 사용
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
encoded_inputs = tokenizer(reviews, max_length=128, padding=True, truncation=True, return_tensors='tf')
```
### ~~3.3 모델 구조 (기존 LSTM)~~

```python
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3개 클래스 분류
])
```


### 3.4 모델 구조 (개선 BERT)
- **사전 학습 모델**: `klue/bert-base` (한국어 특화 BERT 모델)
- **전이 학습 방식**: 사전 학습된 BERT 모델 위에 분류를 위한 Dense 레이어를 추가하여 Fine-tuning(미세 조정)

```python
# CustomBertForSequenceClassification (in Mining.py)

class CustomBertForSequenceClassification(tf.keras.Model):
    def __init__(self, bert_model_core, num_labels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model_core
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(num_labels, name="classifier")

    def call(self, inputs, training=False):
        outputs = self.bert(inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            training=training)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits
```

### 3.5 학습 결과
(BILSTM)
![스크린샷 2025-05-20 123558](https://github.com/user-attachments/assets/44e9c859-bb74-47a4-a680-4d2d89bfb6ef)

총 데이터: 304,027건<br>
테스트 정확도: 88.22% (LSTM), <br>

(BERT)<br>
![image](https://github.com/user-attachments/assets/4e53843d-7696-416a-9c8b-a1114e66da98)

총 데이터: 40만건<br>
테스트 정확도 : 87.34% (BERT)<br>


## 4. 핵심 코드

### 4.1 데이터 수집 (Crawler.py)
### 핵심 기능: 네이버 쇼핑에서 책 검색 후 리뷰 수집
<br>

```python
def find_book(text):

    # 1. 웹 드라이버로 네이버 쇼핑 접속
    driver = webdriver.Chrome()
    url = f"[https://search.shopping.naver.com/book/search?query=](https://search.shopping.naver.com/book/search?query=){text}"

    # 2. 검색 결과에서 책 링크들 추출
    elements = driver.find_elements(By.CSS_SELECTOR, 'a.bookListItem_info_top__DLxpl')
    hrefs = [el.get_attribute('href') for el in elements]

    # 3. 각 책 페이지 방문하여 리뷰 수집
    for detail_url in hrefs:
        # 리뷰 탭 클릭 → 더보기 버튼 클릭 → 리뷰 텍스트 추출
        reviews = soup.select('#book_section-review > ul > li > div.reviewItem_review__LEKrI > p')
```

### 4.2 감성 분석 (Mining.py)
### 핵심 기능: 수집된 리뷰를 BERT 모델로 감성 분석
<br>

```python
def analyze_sentiment(reviews):

    # 1. 저장된 BERT 토크나이저와 학습된 모델 가중치 불러오기
    tokenizer = BertTokenizer.from_pretrained('.')
    model = CustomBertForSequenceClassification(...) # 모델 구조 생성
    model.load_weights('tf_model.h5')

    # 2. 텍스트를 BERT 입력 형식으로 변환 (토큰화)
    encoded_inputs = tokenizer(
        reviews,
        max_length=128,
        padding='max_length',
        return_tensors='tf'
    )

    # 3. 모델로 감성 예측
    predictions = model.predict(dict(encoded_inputs))

    # 4. 결과를 이해하기 쉽게 변환
    sentiment_map = {0: "부정적", 1: "중립적", 2: "긍정적"}
    results = []
    # ... (결과 후처리) ...
```

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
##LSTM 사용시## <br>
![스크린샷 2025-06-24 211252](https://github.com/user-attachments/assets/4daec3a4-1fdf-43e1-bd81-90de72e72f5d)
![스크린샷 2025-06-24 211304](https://github.com/user-attachments/assets/ed4d2392-d84e-4224-a24b-72aebebef44c)
![스크린샷 2025-06-24 211308](https://github.com/user-attachments/assets/414dea0e-291a-4ef5-b74c-452d330791fc)


##BERT 사용시## <br>
![스크린샷 2025-06-24 211838](https://github.com/user-attachments/assets/59ccdcf7-47cd-4121-82c6-b115afdbafa8)
![스크린샷 2025-06-24 211853](https://github.com/user-attachments/assets/2ebad13a-e15b-4b62-bbc4-a192b79780dd)
![스크린샷 2025-06-24 211857](https://github.com/user-attachments/assets/c9d4048c-68ee-412d-b434-04478363c1fa)
![스크린샷 2025-06-24 211902](https://github.com/user-attachments/assets/f026424b-6b05-45c0-adc3-92012b9d7090)



BERT 모델의 감성분석이 좀더 좋게 나온걸 확인 가능합니다
