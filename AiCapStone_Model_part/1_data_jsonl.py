import json
import random

# 1. JSON 파일들 한꺼번에 불러오기
import os

base_dir = os.path.join(os.path.dirname(__file__), 'data')

file_map = {
    'scenarios':      os.path.join(base_dir, 'scenario_definitions.json'),
    'class_defs':     os.path.join(base_dir, 'Class_definitions.json'),
    'emotion_defs':   os.path.join(base_dir, 'Emotion_definitions.json'),
    'age_defs':       os.path.join(base_dir, 'Age_definitions.json'),
    'persona_defs':   os.path.join(base_dir, 'Personality_definitions.json'),
    'era_defs':       os.path.join(base_dir, 'Era_definitions.json'),
    'relation_defs':  os.path.join(base_dir, 'Relation_definitions.json'),
    'role_defs':      os.path.join(base_dir, 'Role_definitions.json'),
    'status_defs':    os.path.join(base_dir, 'Social_status_definitions.json'),
}

for var_name, file_name in file_map.items():
    with open(file_name, 'r', encoding='utf-8') as f:
        globals()[var_name] = json.load(f)

# 2. 클래스 이름 → 역할 매핑 구축
class_to_roles = {}
for cls in class_defs:
    roles_list = []
    if 'subclasses' in cls:
        for subclass in cls['subclasses']:
            roles_list.extend(subclass.get('roles', []))
    else:
        roles_list.extend(cls.get('roles', []))
    class_to_roles[cls['class']] = roles_list

# 3. 감정 예시에 tags 및 intensity 포함하여 저장
emotion_examples = {}
for emo in emotion_defs:
    trait_name = emo.get('trait') or emo.get('emotion')
    if trait_name:
        examples = [
            {
                "text": ex["text"],
                "tags": ex.get("tags", []),
                "intensity": ex.get("intensity", None)
            }
            for ex in emo.get('examples', [])
        ]
        emotion_examples[trait_name] = examples

# 4. JSONL 파일 생성
with open('lora_data.jsonl', 'w', encoding='utf-8') as outfile:
    for scenario in scenarios:
        template           = scenario['template']
        allowed_eras       = scenario.get('allowed_eras', [None])
        allowed_classes    = scenario.get('allowed_classes', [None])
        allowed_roles_spec = scenario.get('allowed_roles', None)
        allowed_status     = scenario.get('allowed_social_status', [None])
        allowed_ages       = scenario.get('allowed_ages', [None])
        allowed_personas   = scenario.get('allowed_personalities', [None])
        allowed_emotions   = scenario.get('allowed_emotions', [])
        allowed_relations  = scenario.get('allowed_relations', [None])

        # 빈 리스트 처리 → 제한 없음
        if allowed_roles_spec is not None and len(allowed_roles_spec) == 0:
            allowed_roles_spec = [None]
        if allowed_status is not None and len(allowed_status) == 0:
            allowed_status = [None]
        if allowed_ages is not None and len(allowed_ages) == 0:
            allowed_ages = [None]
        if allowed_personas is not None and len(allowed_personas) == 0:
            allowed_personas = [None]
        if len(allowed_emotions) == 0:
            continue

        # 역할 결정
        if allowed_roles_spec:
            roles_for_scenario = allowed_roles_spec
        else:
            roles_for_scenario = []
            for cls_name in allowed_classes:
                if cls_name is None:
                    continue
                roles = class_to_roles.get(cls_name, []) or [None]
                roles_for_scenario.extend(roles)
            roles_for_scenario = list(set(roles_for_scenario)) or [None]

        # 모든 조합 반복
        for era in allowed_eras:
            for social in allowed_status:
                for cls_name in allowed_classes:
                    for role in roles_for_scenario:
                        for age in allowed_ages:
                            for persona in allowed_personas:
                                for relation in allowed_relations:
                                    for emo_trait in allowed_emotions:
                                        # 감정 예시 존재 확인
                                        if emo_trait not in emotion_examples or not emotion_examples[emo_trait]:
                                            continue
                                        response_data = random.choice(emotion_examples[emo_trait])

                                        # 캐릭터 설명 생성
                                        desc_parts = []
                                        if era: desc_parts.append(f"{era} 시대의")
                                        if age: desc_parts.append(age)

                                        if social:
                                            if social in ["귀족", "왕족"] and role:
                                                sr = f"{social} 출신 {role}"
                                            elif social in ["신격", "정령"]:
                                                sr = f"{social} 존재" if not role else f"{social} {role}"
                                            else:
                                                sr = f"{social} {role}" if role else social
                                        else:
                                            sr = role or cls_name or ""
                                        if sr: desc_parts.append(sr)

                                        desc_main = " ".join(desc_parts) or "캐릭터"
                                        character_desc = f"이 캐릭터는 {desc_main}입니다."
                                        if persona:
                                            character_desc += f" {persona} 성격을 지녔습니다."

                                        instruction_text = (
                                            f"캐릭터 설명: {character_desc}\n"
                                            f"상황: {template}"
                                        )

                                        # JSONL 출력
                                        entry = {
                                            "instruction": instruction_text,
                                            "response": {
                                                "text": response_data["text"],
                                                "tags": response_data.get("tags", []),
                                                "intensity": response_data.get("intensity", None)
                                            }
                                        }
                                        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
