import os
import sys
import argparse
import json
import time
import requests
from typing import Dict, List, Any
from openai import OpenAI

# =============================================
# 파일 로드 유틸
# =============================================
def load_json(path: str) -> Any:
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def parse_text_file(path: str):
    """
    주어진 텍스트 파일에서 빈 줄을 제외한 각 줄을 문자열 리스트로 반환합니다.
    """
    with open(path, encoding='utf-8') as f:
        return f.read()

# =============================================
# GPT 호출하여 직업 프로필 생성
# =============================================
def generate_profile(
        client: OpenAI,
        job_pair: List[str],
        functions: List[Dict[str, Any]],
        example: str =""
) -> Dict[str, Any]:
    name, origin = job_pair
    prompt = (
        f"직업 '{name}'({origin})에 대한 프로필을 생성해줘. "
        "한국어로 function call 형식에 맞게 리턴해줘."
        "예시는 다음과 같다"
        f"{example}"
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[{"role":"user","content":prompt}],
        functions=functions,
        function_call={"name":"generate_job_profile"}
    )
    args = resp.choices[0].message.function_call.arguments
    return json.loads(args)

# =============================================
# 메인: CLI 파서 및 워크플로우
# =============================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="JSON 역할 파일을 읽어 직업별 프로필을 생성하고 저장합니다."
    )
    parser.add_argument('-i','--id', required=True,
                        help='SRS 서비스에 등록된 사용자 ID')
    parser.add_argument('-f','--input', default='role.json',
                        help='입력 역할 JSON 파일 경로')
    parser.add_argument('-d','--definitions', default='function_definitions.json',
                        help='Function Call 스키마 JSON 파일 경로')
    parser.add_argument('-o','--output', default='output.json',
                        help='결과 JSON 저장 경로')
    args = parser.parse_args()

    # 1) API 키 조회
    resp = requests.get(f"https://srs.jftt.kr/openaikey/{args.id}")
    api_key = resp.text.strip()
    if not api_key:
        print('[Error] 올바른 SRS ID가 아닙니다.')
        sys.exit(1)

    # 2) OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)

    # 3) 함수 스키마 및 역할 데이터 로드
    functions = load_json(args.definitions)
    if isinstance(functions, dict):
        functions = [functions]
    role_data: Dict[str, Dict[str, List[List[str]]]] = load_json(args.input)

    example = parse_text_file('example.json')

    # 4) 프로필 생성
    result: Dict[str, Any] = {}
    for major, subcats in role_data.items():
        for sub, jobs in subcats.items():
            for job_pair in jobs:
                print(f"[Info] {major} → {sub} → {job_pair[0]}({job_pair[1]}) 프로필 생성 중...")
                profile = generate_profile(client, job_pair, functions, example)
                result[job_pair[0]] = profile
                time.sleep(0.3)

    # 5) 결과 저장
    with open(args.output, 'r+', encoding='utf-8') as fout:
        fout.seek(0)           # 파일 포인터를 맨 앞으로 이동
        fout.truncate(0)       # 파일 내용을 전부 지움
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"[Success] 프로필 생성 완료: {args.output}")
