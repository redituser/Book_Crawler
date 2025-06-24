# Visualizer.py에 감성 그래프 함수 추가
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Windows':
    plt.rc('font' , family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
# 기존 함수는 유지

# 감성 분석 파이 차트 (간단 버전)
def show_sentiment_pie(counts):
    """감성 분석 결과를 파이 차트로 표시합니다"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['긍정적', '중립적', '부정적']
    # counts 딕셔너리에 키가 없는 경우 0으로 처리
    values = [counts.get('긍정적', 0), counts.get('중립적', 0), counts.get('부정적', 0)]
    colors = ['green', 'yellow', 'red']
    
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('리뷰 감성 분석 결과')
    
    return fig