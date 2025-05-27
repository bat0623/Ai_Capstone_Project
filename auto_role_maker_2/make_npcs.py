import json
import sys
import copy
import requests
from openai import OpenAI

def _get_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_age(file_path="args_definitions/Age_definitions.json"):
    data_list = _get_json_data(file_path)
    for age_dict in data_list:
        age_group = age_dict["age_group"]
        description = age_dict["description"]
        yield [age_group, description]

def get_era(file_path="args_definitions/Era_definitions.json"):
    data_list = _get_json_data(file_path)
    for era_dict in data_list:
        era = era_dict["era"]
        description = era_dict["description"]
        yield [era, description]

def get_job(file_path="args_definitions/Role_definitions.json"):
    data_dicts = _get_json_data(file_path)
    for role,value in data_dicts.items():
        job = role
        summary = value["summary"]
        yield [job, summary]

def get_social_status(file_path="args_definitions/Social_status_definitions.json"):
    data_list = _get_json_data(file_path)
    for social_status_dict in data_list:
        status = social_status_dict["status"]
        description = social_status_dict["description"]
        yield [status, description]

def get_gender():
    yield "남자"
    yield "여자"
    yield "무성"

def get_relation(file_path="args_definitions/Relation_definitions.json"):
    data_list = _get_json_data(file_path)
    for relation_dict in data_list:
        relation = relation_dict["relation"]
        description = relation_dict["description"]
        yield [relation, description]

def get_city_background(file_path="args_definitions/backgrounds_cities.json"):
    data = _get_json_data(file_path)
    backgrounds_list=data["backgrounds"]
    for background in backgrounds_list:
        background_code=background["code"]
        background_era=background["era"]
        background_description=background["description"]
        for city_dict in background["cities"]:
            city_name=city_dict["name"]
            city_description=city_dict["description"]
            yield [background_code,background_era,background_description,city_name,city_description]

def get_api_key(key_id):
    resp = requests.get(f"https://srs.jftt.kr/openaikey/{key_id}")
    api_key = resp.text.strip()
    if not api_key:
        print('[Error] 올바른 SRS ID가 아닙니다.')
        sys.exit(1)
    return api_key
openai = OpenAI(api_key=get_api_key(input("SRS ID 입력: ")))

def check_coexistence(npc, description_dict):
    """
    npc 속성들의 조합이 논리적으로 모순 없는지 검사.
    맞으면 True, 아니면 False 반환.
    """
    func_def = {
        "name": "check_coexistence",
        "description": "여러 요소들이 논리적으로 공존 가능 여부 및 논리적으로 적합 여부 등을 판단",
        "parameters": {
            "type": "object",
            "properties": {
                "allowed": {
                    "type": "boolean",
                    "description": "모순이 없으면 true, 모순이 있으면 false"
                }
            },
            "required": ["allowed"]
        }
    }
    # system + user 메시지 구성
    messages = [
        {"role": "system", "content":
            "당신은 NPC 속성들의 논리적으로 공존 가능 여부 및 논리적으로 적합 여부를 판단하는 역할입니다. 무엇보다 상식적으로 게임에 적합할 수 있는지 판단하세요. 어색한 요소들이 있을 경우 allowed=false로 답해주세요."
         },
        {"role": "user", "content":
            f"NPC 정보: {json.dumps(npc, ensure_ascii=False)}\n"
            f"설명: {json.dumps(description_dict, ensure_ascii=False)}\n"
            "위 속성들이 함께 있을 때 모순이 없으면 allowed=true, 모순이면 allowed=false로 답해주세요."
         }
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        functions=[func_def],
        function_call={"name": "check_coexistence"},
        temperature=0
    )
    # 반환된 함수 호출 응답에서 allowed 값 꺼내기
    args = resp.choices[0].message.function_call.arguments
    json_args = json.loads(args)
    return json_args.get("allowed", False)


def get_name_description(npc, description_dict):
    """
    npc dict 를 보고 이름(name)과 간단한 설명(description)을 생성.
    """
    func_def = {
        "name": "generate_npc_profile",
        "description": "NPC의 이름과 설명을 생성",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "생성된 NPC 한글 이름, 다양한 이름을 사용할 것"
                },
                "description": {
                    "type": "string",
                    "description": "NPC의 배경을 설명하는 한글 문장, 여러 어젠더를 사용하여 다양하게 할 것"
                }
            },
            "required": ["name", "description"]
        }
    }
    messages = [
        {"role": "system", "content":
            "당신은 주어진 NPC 속성으로 적절한 이름과 설명을 만드는 작가입니다."
         },
        {"role": "user", "content":
            f"NPC 정보: {json.dumps(npc, ensure_ascii=False)}\n"
            f"세부 설명: {json.dumps(description_dict, ensure_ascii=False)}\n"
            "이 NPC에 어울리는 이름과 한 문장 설명을 생성해주세요."
         }
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        functions=[func_def],
        function_call={"name": "generate_npc_profile"},
        temperature=1.3
    )
    args = resp.choices[0].message.function_call.arguments
    json_args = json.loads(args)
    print(json_args)  # 실제 값 확인
    name = json_args.get("name", "이름없음")
    desc = json_args.get("description", "설명 없음")
    return name, desc

def turn_npc_args():
    npc=dict()
    description_dict=dict()

    for city in get_city_background():
        npc["background_code"]=city[0]
        npc["city"]=city[3]
        npc["era"]=city[1]
        description_dict["background"]=city[2]
        description_dict["city"]=city[4]
        for age in get_age():
            npc["age"]=age[0]
            description_dict["age"]=age[1]
            for gender in get_gender():
                npc["gender"]=gender
                for job in get_job():
                    npc["job"]=job[0]
                    description_dict["job"]=job[1]
                    for social_status in get_social_status():
                        npc["social_status"]=social_status[0]
                        description_dict["social_status"]=social_status[1]
                        for relation in get_relation():
                            npc["relation"]=relation[0]
                            description_dict["relation"]=relation[1]
                            yield [copy.deepcopy(npc), copy.deepcopy(description_dict)]



def make_npc():
    code_int=0
    for npc, description_dict in turn_npc_args():
        if not check_coexistence(npc, description_dict):
            print(f"존재할 수 없는 NPC: {npc}")
            continue
        npc["code"]="N"+str(code_int)
        npc["type"]="npc"
        name, desc = get_name_description(npc, description_dict)
        npc["name"] = name
        npc["description"] = desc
        code_int+=1
        print(f"NPC 생성: {npc}")
        yield copy.deepcopy(npc)





if __name__ == "__main__":
    try:
        npcs=list()
        for npc in make_npc():
            npcs.append(npc)
    except KeyboardInterrupt:
        print("사용자가 취소하였습니다.")

    except Exception as e:
        print(f"에러 발생: {e}")

    finally:
        # JSON 파일로 덤프
        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(
                npcs,
                f,
                ensure_ascii=False,  # 한글이 깨지지 않도록
                indent=2             # 보기 좋게 들여쓰기
            )
        print("output.json에 저장 완료.")