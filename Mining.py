# 최종 수정된 Mining.py

import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
import os

# --- 개선점 1: 모델과 토크나이저를 전역적으로 한 번만 로드 ---

MODEL_NAME = 'klue/bert-base'
SAVED_MODEL_PATH = '.' 
MAX_LEN = 128

print("모델과 토크나이저를 로딩합니다... (시간이 걸릴 수 있습니다)")

# 1. 토크나이저 로딩
tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_PATH)

# 2. 모델 구조 생성 및 가중치 로딩
# 2-1. 기본 BERT 모델 로드
bert_core = TFBertModel.from_pretrained(MODEL_NAME)

# 2-2. 사용자 정의 모델 클래스 정의
class CustomBertForSequenceClassification(tf.keras.Model):
    def __init__(self, bert_model_core, num_labels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model_core
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(num_labels, name="classifier")

    def call(self, inputs, training=False):
        outputs = self.bert(inputs['input_ids'], attention_mask=inputs['attention_mask'], training=training)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits

# 2-3. 최종 모델 인스턴스 생성
model = CustomBertForSequenceClassification(bert_core, num_labels=3)

# 2-4. 모델 빌드를 위한 더미 입력 전달
dummy_inputs = {
    'input_ids': tf.zeros((1, MAX_LEN), dtype=tf.int32),
    'attention_mask': tf.zeros((1, MAX_LEN), dtype=tf.int32)
}
_ = model(dummy_inputs)

# 2-5. 저장된 가중치 로드 (os.path.join 사용)
weights_path = os.path.join(SAVED_MODEL_PATH, 'tf_model.h5')
model.load_weights(weights_path)

print("모델과 토크나이저 로딩 완료.")


# --- analyze_sentiment 함수는 이제 분석 로직에만 집중 ---
def analyze_sentiment(reviews):
    """미리 로드된 BERT 모델과 토크나이저를 사용하여 리뷰 감성을 분석합니다."""
    
    # 리뷰 전처리 및 토큰화
    encoded_inputs = tokenizer(
        reviews,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='tf'
    )
    
    # 감성 예측
    predictions = model.predict(dict(encoded_inputs))
    probabilities = tf.nn.softmax(predictions, axis=-1).numpy()
    
    # 결과 정리
    sentiment_map = {0: "부정적", 1: "중립적", 2: "긍정적"}
    results = []
    
    for i, review_text in enumerate(reviews):
        sentiment_idx = np.argmax(probabilities[i])
        sentiment = sentiment_map[sentiment_idx]
        confidence = float(probabilities[i][sentiment_idx])
        
        results.append({
            'text': review_text,
            'sentiment': sentiment,
            'confidence': confidence
        })
        
    # 감성 카운트
    sentiments = [r['sentiment'] for r in results]
    counts = {
        '부정적': sentiments.count('부정적'),
        '중립적': sentiments.count('중립적'),
        '긍정적': sentiments.count('긍정적')
    }
    
    return {
        'reviews': results,
        'counts': counts,
        'total': len(results)
    }