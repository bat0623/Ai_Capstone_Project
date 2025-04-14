import openai
import json
from request_srs import get_openaikey

client = openai.OpenAI(api_key=get_openaikey("bat0623",input("비멀번호: ")))  # 자신의 실제 API키로 교체하세요.


def request_openai_task(prompt, model="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    """
    OpenAI 모델에 작업을 요청하는 메서드입니다.

    Parameters:
        prompt (str): 요청할 작업 또는 질문 내용.
        model (str): 사용할 OpenAI 모델 (기본값: gpt-3.5-turbo).
        max_tokens (int): 최대 생성 토큰 수 (기본값: 150).
        temperature (float): 창의성 지수(0에서 2 사이의 값, 기본값: 0.7).

    Returns:
        str: OpenAI 모델이 생성한 결과 텍스트.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        # 응답 메시지 추출
        message = response.choices[0].message.content.strip()

        return message

    except openai.OpenAIError as e:
        print(f"OpenAI API 요청 실패: {e}")
        return None


# OpenAI Function Call 요청 메서드
def request_openai_func_call(user_prompt):
    # 함수 정의 (계산기 예시)
    functions = [
        {
            "name": "calculator",
            "description": "사칙연산 계산을 수행함",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {
                        "type": "number",
                        "description": "첫번째 숫자"
                    },
                    "num2": {
                        "type": "number",
                        "description": "두번째 숫자"
                    },
                    "operation": {
                        "type": "string",
                        "description": "수행할 계산 연산",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    }
                },
                "required": ["num1", "num2", "operation"]
            }
        }
    ]

    try:
        # API에 요청을 보냄 (새로운 포맷에 맞춤)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{"type": "function", "function": func} for func in functions],
            tool_choice="auto"
        )

        message = response.choices[0].message

        # 함수 호출 결과인지 확인
        if message.tool_calls:
            func_name = message.tool_calls[0].function.name
            func_args = json.loads(message.tool_calls[0].function.arguments)

            print(f"모델이 호출한 함수: {func_name}")
            print(f"전달된 인자: {func_args}")

            # 실제 함수 실행 (계산기 함수 실행 예시)
            if func_name == "calculator":
                result = calculator(**func_args)
                return result

        return message.content

    except openai.OpenAIError as e:
        print(f"OpenAI API 요청 실패: {e}")
        return None


# 실제 실행할 함수 정의 (계산기 예시)
def calculator(num1, num2, operation):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2 if num2 != 0 else "0으로 나눌 수 없습니다."
    else:
        return "알 수 없는 연산입니다."


if __name__ == '__main__':
    prompt_example = "Python으로 간단한 hello world 프로그램을 작성해줘."
    response = request_openai_task(prompt_example)

    if response:
        print("OpenAI의 응답 결과:")
        print(response)
    else:
        print("OpenAI 작업 요청 중 오류가 발생했습니다.")

    prompt_example = "32에서 8을 나누어 줘."
    result = request_openai_func_call(prompt_example)

    print("결과:", result)
