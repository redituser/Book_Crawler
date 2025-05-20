import streamlit as st
import Crawler as nbc
import Mining as mining
import Visualizer as viz

# 검색 기능
st.title("책 리뷰 감성 분석")
keyword = st.sidebar.text_input("검색 키워드")

if keyword:
    # 진행 상황 표시
    with st.spinner(f"'{keyword}' 관련 책 및 리뷰 검색 중..."):
        reviews = nbc.find_book(keyword)
    
    if reviews and len(reviews) > 0:
        st.success(f"총 {len(reviews)}개의 리뷰를 찾았습니다!")
        
        # 감성 분석 진행
        with st.spinner("리뷰 감성 분석 중..."):
            analysis = mining.analyze_sentiment(reviews)
        
        # 결과 표시
        st.header("         감성 분석 결과")
        
        # 파이 차트 표시
        chart = viz.show_sentiment_pie(analysis['counts'])
        st.pyplot(chart)
        
        # 간단한 통계 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("긍정적 리뷰", f"{analysis['counts']['긍정적']}개", 
                   f"{analysis['counts']['긍정적']/analysis['total']*100:.1f}%")
        col2.metric("중립적 리뷰", f"{analysis['counts']['중립적']}개", 
                   f"{analysis['counts']['중립적']/analysis['total']*100:.1f}%")
        col3.metric("부정적 리뷰", f"{analysis['counts']['부정적']}개", 
                   f"{analysis['counts']['부정적']/analysis['total']*100:.1f}%")
        
        # 리뷰 샘플 표시
        with st.expander("일부 리뷰 예시 보기"):
            # 각 감성별 리뷰 샘플 3개씩 표시
            for sentiment in ["긍정적", "중립적", "부정적"]:
                samples = [r for r in analysis['reviews'] if r['sentiment'] == sentiment][:3]
                if samples:
                    st.subheader(f"{sentiment} 리뷰 예시")
                    for i, sample in enumerate(samples, 1):
                        st.markdown(f"**{i}. {sample['text']}**")
                        st.caption(f"확신도: {sample['confidence']*100:.1f}%")
                    st.markdown("---")
    else:
        st.warning("리뷰를 찾을 수 없습니다. 다른 키워드로 검색해보세요.")