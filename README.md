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



## 3. 모델 학습 방법 & 결과
https://github.com/redituser/Book_Review_SA_Model (참고)


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



---

## 5. 실행 방법

### 5.1 사전 준비

먼저 아래와 같은 환경이 준비되어 있어야 합니다:

* Python 3.8 이상
* 필요한 라이브러리 설치:

```bash
pip install -r requirements.txt
```

또는 아래 개별 설치:

```bash
pip install streamlit transformers tensorflow selenium beautifulsoup4 matplotlib
```

---

### 5.2 실행 방법 (Streamlit WebApp)

1. **터미널/명령 프롬프트 열기**
2. `Main.py` 파일이 있는 디렉토리로 이동:

```bash
cd /자신의/프로젝트/경로
```

예시:

```bash
cd C:\Users\myname\projects\Book_Review_SA
```

3. **Streamlit 앱 실행**:

```bash
streamlit run Main.py
```

4. 브라우저가 자동으로 열리며 웹 UI에서 감성 분석 사용 가능
   (열리지 않으면 `http://localhost:8501` 직접 입력)

---

### 5.3 실행 시 주의 사항

* `ChromeDriver`와 크롬 브라우저 버전이 맞아야 합니다.
* 크롤링 시 사이트 구조 변경에 따라 코드 수정이 필요할 수 있습니다.
* BERT 모델 학습이 완료된 상태여야 정상 동작합니다 (`tf_model.h5` 필요).

