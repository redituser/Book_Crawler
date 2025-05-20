# Visualizer.py에 감성 그래프 함수 추가
import matplotlib.pyplot as plt

# 기존 함수는 유지

# 감성 분석 파이 차트 (간단 버전)
def show_sentiment_pie(counts):
    """감성 분석 결과를 파이 차트로 표시합니다"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ['red', 'yellow', 'green']
    
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
    ax.set_title('리뷰 감성 분석 결과')
    
    return fig