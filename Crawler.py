
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time


def find_book (text):

    # 1) 드라이버 실행
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)

    # 2) 검색 페이지 열기
    url = f"https://search.shopping.naver.com/book/search?query={text}" 
    driver.get(url)

    # 3) 페이지가 렌더링될 때까지 대기
    time.sleep(3)

    # 4) 동적으로 로드된 <a> 요소들 찾기 (책 링크들)
    elements = driver.find_elements(By.CSS_SELECTOR, 'a.bookListItem_info_top__DLxpl')

    # 5) href만 뽑아내기
    hrefs = [el.get_attribute('href') for el in elements]
    print(f"총 {len(hrefs)}개의 책 링크를 찾았습니다.")

    # 모든 리뷰를 저장할 리스트
    all_reviews = []

    all_items = []
    book_titles = [] 
        


    # 각 책 페이지 방문
    for i, detail_url in enumerate(hrefs, 1):
        
        
        
        print(f"\n{i}번째 책 페이지 방문 중: {detail_url}")
        driver.get(detail_url)
        time.sleep(2)  # 페이지 로딩 대기

        #타이틀 추출 
        try:
            title = driver.find_element(By.CSS_SELECTOR, 'h2.bookTitle_book_name__iDaSl')
            title_text = title.text
            book_titles.append(title_text)
            all_items.append(title_text)
            if title is None:  # 'in None'이 아니라 'is None'으로 수정
                print('제목이 비어있음')
                continue
        except Exception as e:  # 첫 번째 try 블록에 대한 except 추가
            print(f"제목 추출 중 오류 발생: {e}")
            continue  # 오류 발생 시 다음 책으로 넘어감

        
        # 리뷰 탭 찾기 및 클릭
        try:
            # 여러 가능한 리뷰 탭 선택자 시도
            review_tabs = driver.find_elements(By.CSS_SELECTOR, "a.reviewTab_btn_tab__b8bww")
            if not review_tabs:
                review_tabs = driver.find_elements(By.CSS_SELECTOR, "a.pcFixedTab_btn_tab__8amCI")
                
            
            # 책 리뷰 탭 찾기
            clicked = False
            for tab in review_tabs:
                if "책 리뷰" in tab.text or "도서몰" in tab.text or "리뷰" in tab.text:
                    tab.click()
                    print("리뷰 탭 클릭 성공")
                    time.sleep(2)  # 탭 전환 대기
                    clicked = True
                    break
            
            if not clicked:
                print("리뷰 탭을 찾을 수 없습니다.")
                continue
                
            # 리뷰 더보기 버튼이 있는 경우 계속 클릭
            more_reviews_clicked = 0
            while True:
                try:
                    # 여러 가능한 더보기 버튼 선택자 시도
                    more_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "#book_section-review > a"))
                    )
                    
                    # 더보기 버튼이 보이면 클릭
                    if more_button.is_displayed():
                        more_button.click()
                        more_reviews_clicked += 1
                        print(f"더보기 버튼 {more_reviews_clicked}번 클릭")
                        time.sleep(1.5)  # 추가 리뷰 로딩 대기
                    else:
                        break
                except (TimeoutException, NoSuchElementException):
                    print("더 이상 더보기 버튼이 없습니다.")
                    break
                
            # 모든 리뷰 추출
            soup = BeautifulSoup(driver.page_source, 'html.parser')

        
            
            # 여러 가능한 리뷰 선택자 시도
            reviews = soup.select('#book_section-review > ul > li > div.reviewItem_review__LEKrI > p')
            if not reviews:
                reviews = soup.select('div.reviewItem_text__3QGbv')
            if not reviews:
                reviews = soup.select('p.reviewItem_text_area__hCPSh')
                
            book_reviews = []
            
            for idx, review in enumerate(reviews, 1):
                review_text = review.get_text(strip=True)
                if review_text:  # 빈 리뷰 제외
                    book_reviews.append(review_text)
                    print(f"리뷰 {idx}: {review_text[:50]}...")  # 리뷰 앞부분만 출력
            
            print(f"총 {len(book_reviews)}개의 리뷰를 추출했습니다.")
            all_items.extend(book_reviews)
            
        except Exception as e:
            print(f"오류 발생: {e}")
            continue

    print(f"\n모든 책에서 총 {len(all_reviews)}개의 리뷰를 수집했습니다.")
    return all_items



    driver.quit()