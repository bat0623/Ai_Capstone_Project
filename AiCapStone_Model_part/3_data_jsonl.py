import os
import json
from itertools import product

# ------------------------------------------------------------------
# 옵션: True면 각 감정별 모든 예시 문장을, False면 첫 번째 예시만 JSONL에 포함합니다.
USE_ALL_EMOTION_EXAMPLES = True

# ------------------------------------------------------------------
# 기본 경로 설정: 이 스크립트 파일(__file__) 기준으로 'data' 폴더를 가리킵니다.
base_dir = os.path.join(os.path.dirname(__file__), 'data')

# 정의 파일들의 절대 경로를 한곳에 정리
file_map = {
    'scenarios':       os.path.join(base_dir, 'scenario_definitions.json'),
    'classes':         os.path.join(base_dir, 'Class_definitions.json'),
    'personalities':   os.path.join(base_dir, 'Personality_definitions.json'),
    'emotions':        os.path.join(base_dir, 'Emotion_definitions.json'),
    'ages':            os.path.join(base_dir, 'Age_definitions.json'),
    'eras':            os.path.join(base_dir, 'Era_definitions.json'),
    'relations':       os.path.join(base_dir, 'Relation_definitions.json'),
    'social_statuses': os.path.join(base_dir, 'Social_status_definitions.json'),
}

# ------------------------------------------------------------------
# 1) JSON 파일 로드
with open(file_map['scenarios'], 'r', encoding='utf-8') as f:
    scenarios = json.load(f)
with open(file_map['classes'], 'r', encoding='utf-8') as f:
    classes = json.load(f)
with open(file_map['personalities'], 'r', encoding='utf-8') as f:
    personalities = json.load(f)
with open(file_map['emotions'], 'r', encoding='utf-8') as f:
    emotions = json.load(f)
with open(file_map['ages'], 'r', encoding='utf-8') as f:
    ages = json.load(f)
with open(file_map['eras'], 'r', encoding='utf-8') as f:
    eras = json.load(f)
with open(file_map['relations'], 'r', encoding='utf-8') as f:
    relations = json.load(f)
with open(file_map['social_statuses'], 'r', encoding='utf-8') as f:
    social_statuses = json.load(f)

# ------------------------------------------------------------------
# 2) 클래스 이름 → 역할 목록 매핑 생성
class_to_roles = {}
for cls in classes:
    class_name = cls.get("class")
    roles_list = []
    # subclasses가 있으면 그 안의 roles를 모두 모음
    if "subclasses" in cls:
        for sub in cls["subclasses"]:
            roles_list.extend(sub.get("roles", []))
    # 직접 roles키가 있으면 추가
    if "roles" in cls:
        roles_list.extend(cls.get("roles", []))
    # 중복 제거 후 저장
    class_to_roles[class_name] = list(dict.fromkeys(roles_list))

# ------------------------------------------------------------------
# 3) 감정 정의를 trait 이름으로 빠르게 조회할 수 있는 맵 생성
emotion_map = { emo["trait"]: emo for emo in emotions }

# ------------------------------------------------------------------
# 4) 정의된 social status 목록을 집합으로 저장
social_status_set = { status["status"] for status in social_statuses }

# ------------------------------------------------------------------
# 5) class_name과 social_status_hint를 기반으로 역할 목록 반환 함수
def get_roles_for_class(class_name, social_status_filter=None):
    """
    class_name이 None이면 [None]을 반환합니다.
    social_status_filter가 알려진 status가 아니면 role 필터로 해석하여
    해당 문자열을 포함하는 역할만 반환합니다.
    """
    if class_name is None:
        return [None]
    roles = class_to_roles.get(class_name, [])
    # social_status_filter가 정의된 상태가 아니면 역할 필터로 처리
    if social_status_filter and social_status_filter not in social_status_set:
        # 정확히 일치하는 역할이 있으면 그 하나만 반환
        if social_status_filter in roles:
            return [social_status_filter]
        # 포함 관계로 필터링
        filtered = [r for r in roles if social_status_filter in r]
        if filtered:
            return filtered
    return roles

# ------------------------------------------------------------------
# 6) output JSONL 파일 작성 준비
output_path = os.path.join(base_dir, 'lora_data1.jsonl')
with open(output_path, 'w', encoding='utf-8') as outfile:

    # 각 시나리오 정의마다 조합 생성
    for scenario in scenarios:
        template           = scenario.get("template", "")
        allowed_classes    = scenario.get("allowed_classes", [None])
        allowed_statuses   = scenario.get("allowed_social_status", [None])
        allowed_personalities = scenario.get("allowed_personalities", [])
        allowed_emotions   = scenario.get("allowed_emotions", [])
        allowed_relations  = scenario.get("allowed_relations", [None])
        allowed_eras       = scenario.get("allowed_eras", [])
        allowed_ages       = scenario.get("allowed_ages", [None])

        # 모든 속성 조합 생성
        for era in allowed_eras:
            for age in allowed_ages:
                for social_status in allowed_statuses:
                    for class_name in allowed_classes:
                        # 역할 목록 결정 (class + social_status hint)
                        roles_list = get_roles_for_class(class_name, social_status)
                        for role in roles_list:
                            for personality in allowed_personalities:
                                for emotion_trait in allowed_emotions:
                                    emo_def = emotion_map.get(emotion_trait)
                                    if not emo_def:
                                        continue
                                    examples = emo_def.get("examples", [])
                                    # 예시 문장 선택
                                    if USE_ALL_EMOTION_EXAMPLES:
                                        selected = examples
                                    else:
                                        selected = examples[:1]

                                    # 각 예시 문장마다 JSONL 엔트리 작성
                                    for ex in selected:
                                        # instruction 문자열 구성
                                        instr = f"캐릭터 설명: 이 캐릭터는 {era} 시대의 "
                                        if age:
                                            instr += f"{age} "
                                        if class_name is None:
                                            if social_status:
                                                instr += f"{social_status} 존재입니다. "
                                            else:
                                                instr += "정체불명 존재입니다. "
                                        else:
                                            if social_status:
                                                instr += f"{social_status} 출신 "
                                            role_name = role if role is not None else class_name
                                            instr += f"{role_name}입니다. "
                                        instr += f"{personality} 성격을 지녔습니다.\n"
                                        instr += f"상황: {template}"

                                        # response 객체 준비
                                        response_obj = {
                                            "text": ex["text"],
                                            "tags": ex.get("tags", []),
                                            "intensity": ex.get("intensity")
                                        }

                                        # 최종 엔트리 작성 및 파일에 기록
                                        entry = {
                                            "instruction": instr,
                                            "response": response_obj
                                        }
                                        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"완료: '{output_path}'에 JSONL 데이터 생성됨.")
