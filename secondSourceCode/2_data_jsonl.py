import os
import json
from itertools import product

# ----------------- 설정 -----------------
# 현재 스크립트 파일(__file__) 기준으로 data 디렉토리 경로 설정
base_dir = os.path.join(os.path.dirname(__file__), 'data')

# 각 정의 파일의 절대 경로를 맵핑
file_map = {
    'scenarios':      os.path.join(base_dir, 'scenario_definitions.json'),
    'role_defs':      os.path.join(base_dir, 'Role_definitions.json'),
    'class_defs':     os.path.join(base_dir, 'Class_definitions.json'),
    'emotion_defs':   os.path.join(base_dir, 'Emotion_definitions.json'),
    'persona_defs':   os.path.join(base_dir, 'Personality_definitions.json'),
    'era_defs':       os.path.join(base_dir, 'Era_definitions.json'),
    'age_defs':       os.path.join(base_dir, 'Age_definitions.json'),
    'status_defs':    os.path.join(base_dir, 'Social_status_definitions.json'),
    'relation_defs':  os.path.join(base_dir, 'Relation_definitions.json'),
}

# ----------------- JSON 데이터 로드 -----------------
data = {}
for key, path in file_map.items():
    with open(path, 'r', encoding='utf-8') as f:
        data[key] = json.load(f)

# 개별 카테고리 변수로 접근하기 쉽게 할당
scenarios        = data['scenarios']
role_defs        = data['role_defs']
class_defs       = data['class_defs']
emotion_defs     = data['emotion_defs']
personality_defs = data['persona_defs']
era_defs         = data['era_defs']
age_defs         = data['age_defs']
status_defs      = data['status_defs']
relation_defs    = data['relation_defs']

# ----------------- 감정 예시 lookup 테이블 -----------------
# emotion trait 별로 예시 리스트 저장
emotion_examples = {
    entry['trait']: entry.get('examples', [])
    for entry in emotion_defs
}

# ----------------- 성격 trait → 형용사 매핑 -----------------
personality_to_desc = {}
for entry in personality_defs:
    trait = entry['trait']
    if trait.endswith('함'):
        # e.g. "침착함" -> "침착한"
        personality_to_desc[trait] = trait[:-1] + '한'
    elif trait.endswith('적'):
        # e.g. "중립적" -> "중립적인"
        personality_to_desc[trait] = trait + '인'
    elif trait == '자신감':
        personality_to_desc[trait] = '자신감 있는'
    elif trait == '장난기 많음':
        personality_to_desc[trait] = '장난기 많은'
    elif trait == '모험심':
        personality_to_desc[trait] = '모험심 강한'
    elif trait == '호기심':
        personality_to_desc[trait] = '호기심 많은'
    elif trait == '정의감':
        personality_to_desc[trait] = '정의감이 강한'
    elif trait == '야망':
        personality_to_desc[trait] = '야망 있는'
    elif trait == '충성심':
        personality_to_desc[trait] = '충성심 강한'
    elif trait == '복수심':
        personality_to_desc[trait] = '복수심 강한'
    elif trait == '감사하는':
        personality_to_desc[trait] = '감사하는'
    elif trait == '명예감':
        personality_to_desc[trait] = '명예심 강한'
    elif trait == '카리스마':
        personality_to_desc[trait] = '카리스마 있는'
    elif trait == '전투광':
        personality_to_desc[trait] = '전투광적인'
    elif trait == '수집광':
        personality_to_desc[trait] = '수집광적인'
    elif trait == '퀘스트 지향':
        personality_to_desc[trait] = '퀘스트 지향적인'
    elif trait == '행운추구':
        personality_to_desc[trait] = '행운추구적인'
    elif trait == '트릭스터':
        personality_to_desc[trait] = '트릭스터 기질의'
    elif trait == '리더십':
        personality_to_desc[trait] = '리더십 있는'
    else:
        # 그 외는 그대로 사용
        personality_to_desc[trait] = trait

# ----------------- 클래스별 role 분류 -----------------
class_map = {
    "전투 계열": {
        '견습기사', '궁수', '기사', '기술병', '대포병', '도적', '돌격병', '머스킷병',
        '병사', '석궁병', '성기사', '암살자', '용병', '전사', '정찰병', '포탑기사',
        '호위병', '장군', '사령관', '대위', '근위대', '경비원', '문지기', '저격수',
        '총잡이', '돌격대', '스카우트', '팔라딘', '템플러', '대장', '검사'
    },
    "마법·신성 계열": {
        '구속술사', '드루이드', '마법사', '사령술사', '사제', '연금술사', '원소술사',
        '정령술사', '흑마법사', '위저드', '소서러', '워락', '아크메이지', '룬메이지',
        '셰이퍼', '네크로맨서', '블러드워커'
    },
    "지원·치유 계열": {
        '치유사', '음유시인', '힐러', '버프마스터', '축복사', '디버퍼', '바이오리스트'
    },
    "제작·상업 계열": {
        '가죽장이', '대장장이', '상인', '방어구제작자', '재봉사', '목수',
        '건축가', '무역상', '경매사', '돈주인', '길드마스터'
    },
    "사회·귀족 계열": {
        '왕', '여왕', '왕자', '공주', '황제', '황후', '황태자', '황녀',
        '공작', '여공작', '후작', '백작', '백작부인', '자작', '남작',
        '대신', '섭정', '영주', '부족장', '장관', '후계자', '재상', '참사', '조직장'
    },
    "전문·기타 계열": {
        '첩보원', '스파이', '학자', '현자', '사서', '서기', '서기관', '외교관',
        '악사', '무희', '곡예사', '연극배우', '광대', '동료', '파티원', '가이드',
        '안내자', '탐험가', '모험가', '촌장', '함정전문가', '영감', '역사연구자'
    }
}
# 겹치는 역할은 '전문·기타 계열'에서 제거
overlaps = {'후계자', '재상', '참사', '조직장', '부족장'}
for role in overlaps:
    class_map["전문·기타 계열"].discard(role)

# ----------------- 상태(status) 추론 함수 -----------------
royalty_roles   = {'왕','여왕','왕자','공주','황제','황후','황태자','황녀'}
nobility_roles  = {
    '공작', '여공작', '후작', '백작', '백작부인', '자작', '남작',
    '재상', '대신', '섭정', '영주', '장관', '후계자', '조직장', '부족장', '참사'
}
merchant_roles  = {'상인','무역상','경매사','돈주인','길드마스터'}

def infer_status_from_role(role):
    """role 이름으로부터 social status를 추론"""
    statuses = [s['status'] for s in status_defs]
    if role in statuses:
        return role
    if role in royalty_roles:
        return '왕족'
    if role in nobility_roles:
        return '귀족'
    if role in merchant_roles:
        return '상인'
    if role in class_map.get("전투 계열", []):
        return '병사'
    return '시민'

# ----------------- JSONL 데이터 생성 -----------------
jsonl_data = []
for scenario in scenarios:
    template = scenario['template']
    # 시나리오에서 허용하는 값들, 없으면 전체 리스트 사용
    allowed_classes      = scenario.get('allowed_classes',    list(class_map.keys()))
    allowed_statuses     = scenario.get('allowed_social_status',[s['status'] for s in status_defs])
    allowed_personalities= scenario.get('allowed_personalities',[p['trait'] for p in personality_defs])
    allowed_emotions     = scenario.get('allowed_emotions',    [e['trait'] for e in emotion_defs])
    allowed_relations    = scenario.get('allowed_relations',   [r['relation'] for r in relation_defs])
    allowed_eras         = scenario.get('allowed_eras',        [e['era'] for e in era_defs])
    allowed_ages         = scenario.get('allowed_ages',        [a['age_group'] for a in age_defs])

    for era, age, pers, rel, emo in product(
        allowed_eras, allowed_ages,
        allowed_personalities, allowed_relations, allowed_emotions
    ):
        for cls in allowed_classes:
            if cls not in class_map:
                continue
            for status in allowed_statuses:
                # 해당 class+status에 맞는 roles 필터링
                roles = [
                    r for r in class_map[cls]
                    if infer_status_from_role(r) == status
                ]
                if not roles:
                    continue
                for role in roles:
                    # 캐릭터 설명 조합
                    status_part = f"{status} 신분의"
                    age_part    = "" if age == '불명' else f"{age}이고 "
                    pers_desc   = personality_to_desc.get(pers, pers)
                    rel_kor     = {'ally':'동료','enemy':'적대','neutral':'중립'}.get(rel, rel)
                    char_desc = (
                        f"{era} 시대의 {status_part} {cls} {role}, "
                        f"{age_part}{pers_desc} 성격의 캐릭터로 플레이어와 {rel_kor} 관계입니다."
                    )
                    instruction = f"캐릭터 설명: {char_desc} 상황: {template}"

                    # 감정 예시 선택
                    ex_list = emotion_examples.get(emo, [])
                    if ex_list:
                        ex = ex_list[0]
                        response = {
                            "text": ex["text"],
                            "tags": ex.get("tags", []),
                            "intensity": ex.get("intensity")
                        }
                        jsonl_data.append({
                            "instruction": instruction,
                            "response": response
                        })

# ----------------- 결과를 output.jsonl로 저장 -----------------
out_path = os.path.join(base_dir, 'output.jsonl')
with open(out_path, 'w', encoding='utf-8') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"완료: {len(jsonl_data)}건의 항목을 '{out_path}'에 저장했습니다.")
