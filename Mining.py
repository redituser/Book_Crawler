# Mining.py - 간단한 감성 분석 기능
import tensorflow as tf
import numpy as np
import pickle
from keras.utils import pad_sequences
from collections import Counter

def analyze_sentiment(reviews):
    """리뷰 텍스트 목록을 받아 감성 분석 결과를 반환합니다"""
    
    # 1. 모델과 토크나이저 불러오기
    model = tf.keras.models.load_model('sentiment_analysis_model.h5')
    
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('model_config.pickle', 'rb') as f:

        config = pickle.load(f)
    
    max_sequence_len = config['max_sequence_len']
    
    # 2. 리뷰 전처리 및 토큰화
    reviews = [str(review) for review in reviews if review]
    sequences = tokenizer.texts_to_sequences(reviews)
    
    # 3. 패딩 - 모든 리뷰 길이 통일
    padded_data = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')
    
    # 4. 감성 예측
    predictions = model.predict(padded_data)
    
    # 5. 결과 정리
    sentiment_map = {0: "부정적", 1: "중립적", 2: "긍정적"}
    results = []
    
    for i, pred in enumerate(predictions):
        sentiment_idx = np.argmax(pred)
        sentiment = sentiment_map[sentiment_idx]
        confidence = float(pred[sentiment_idx])
        
        results.append({
            'text': reviews[i][:100] + ('...' if len(reviews[i]) > 100 else ''),
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    # 6. 감성 카운트
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