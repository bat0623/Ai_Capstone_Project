import requests

# GET 요청 (예시 URL)
def request_srs_to_text(filename, key, password, verify=True):
    response = requests.get(f"https://srs.jftt.kr/{filename}/{key}?password={password}", verify=verify)
    print("응답 상태코드:", response.status_code)
    return response.text

def get_openaikey(key,password):
    return request_srs_to_text("openaikey", key, password)

if __name__ == "__main__":
    # 내용 출력 (JSON 형식일 경우 .json() 메소드 활용 가능)
    print("응답 데이터:", request_srs_to_text("srstest","testdata","deu_jhlee_pc"))
