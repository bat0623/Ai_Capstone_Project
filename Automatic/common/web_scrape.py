import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def scrape_website(url):
    try:
        # HTTP 요청으로 웹 페이지 내용을 가져옴
        response = requests.get(url)
        response.raise_for_status()  # 요청 실패하면 예외 발생

        # BeautifulSoup으로 웹 페이지 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # 원하는 컨텐츠 추출 예시:
        # 페이지의 모든 링크(a 태그)를 찾아서 반환
        links = []
        for link in soup.find_all('a', href=True):
            links.append(link['href'])

        # 결과 반환
        return links

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []



def scrape_dynamic_website(url):
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 일종의 백그라운드 실행모드 (브라우저가 표시되지 않음)
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(url)

        # 정해진 조건까지 기다림 (예 : 특정 요소 로딩 완료될 때까지 대기)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.content')))

        # 페이지 전체 소스 코드 가져오기
        rendered_html = driver.page_source

        print("Rendered HTML fetched successfully.")

        # 더나아가 BeautifulSoup 으로 파싱까지 한다면
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(rendered_html, 'html.parser')

        # 예제 : 특정 요소의 텍스트 가져오기 ('div.content' 내 모든 텍스트 추출)
        content_div = soup.select_one('div.content')
        if content_div:
            content_text = content_div.get_text(strip=True)
            return content_text
        else:
            return None

    except Exception as e:
        print(f"Scraping failed: {e}")
        return None

    finally:
        # 무조건 드라이버 종료
        driver.quit()

if __name__ == '__main__':
    url = 'https://example.com'  # 스크래핑할 사이트 주소
    links = scrape_website(url)

    # 추출한 링크 출력
    print("Links found:")
    for link in links:
        print(link)
