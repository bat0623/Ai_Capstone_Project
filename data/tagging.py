from pathlib import Path
import json

# 1. 입력 JSON 로드
input_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_ready.json")   # 필요에 따라 경로를 수정하세요.
output_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_final.json")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 감정 및 성격 키워드 사전 정의 (어간 및 활용형 확장)
# Emotion (감정) 키워드 사전: 감정 태그별로 다양한 표현 포함
emotion_keywords = {
    "기쁨": [
        "우와", "멋져", "즐거워", "해냈어", "좋아", "기대돼", "신나", "행복해", "기뻐", 
        "완전 좋아", "웃겨", "즐겁다", "행복", "반가워", "다행", "재밌어", "정말 좋아", 
        "기쁘", "웃고", "재밌", "히히", "헤헤", "축제", "정말 좋다", "기분 최고", 
        "완전 행복", "좋아서", "신난다", "즐겁네", "정말", "멋진 이름이네", "만나서 반가워", 
        "정말 다정하구나", "정말 즐거워", "재밌어 보인다", "기다려줘서 고마워", "신사구나", 
        "안녕하세요"
    ],
    "슬픔": [
        "외로워", "남지 않았어", "미안해", "잘했어야", "눈물이", "그리워", "슬퍼", "괴로워", 
        "절망", "상실감", "울고 싶어", "우울", "비참", "포기", "죽고", "아파", "울", 
        "슬펐어", "아야", "아프잖아", "잃어버렸어", "없어졌어", "잊혀졌어", "외롭다", "속상해", 
        "마음이 아파", "눈물이 나", "그리운", "우울해", "펜던트"
    ],
    "분노": [
        "그만해", "무시하는", "끝장이다", "짜증", "화나", "열받아", "못 참아", "분노", "화가", 
        "겁도 없이", "짜증나", "너 때문", "도저히", "꺼져", "왜 그래", "진짜", "귀찮", "열받", 
        "어이", "뭐하는 거야", "화났어", "열받네", "지금 장난해?", "진짜 왜 이래", "화가 치민다", 
        "끌고 가지 마", "애도 아니고 납치범처럼", "뭐야", "왜 이래"
    ],
    "놀람": [
        "어?!", "뭐야", "세상에", "정말 그런 일이", "이럴 수가", "예상 못 했어", "헉", "말도 안돼", 
        "진짜?", "충격이야", "놀라", "뭐라고", "정말?", "어떻게", "대박", "헐", "어라?", 
        "뭐야 이게", "예상 밖이야", "경쟁심이 강하구나"
    ],
    "두려움": [
        "무서워", "나가고 싶어", "다가오지 마", "죽는 거", "도망치고", "겁이 나", "두렵다", "공포", 
        "불길해", "위험해", "두려워", "불안", "떨려", "망설여", "긴장", "망했다", "혼자 남겨졌어", 
        "어떻게 하지", "불안정해", "여기서 아는 사람도 없어"
    ],
    "불안": [
        "괜찮을까", "불안해", "느낌이 안 좋아", "잘못될", "초조해", "긴장돼", "찝찝해", "걱정돼", 
        "불편해", "위험할지도", "그런 거면 어떡하지", "위험할지도 몰라", "잃어버린 건 아니겠지", 
        "없단 말이야", "제발", "부탁이야", "잃어버린"
    ],
    "혐오": [
        "역겨워", "대기 싫어", "가까이 오지 마", "징그러", "기분 나빠", "혐오스럽다", "불쾌해", 
        "더러워", "냄새나", "지긋지긋", "진절머리나"
    ],
    "당황": [
        "어쩌지", "지금 뭐 하지", "예상 못 했는데", "당황했어", "머리가 하얘졌어", "무슨 말", 
        "혼란스러워", "어라", "말이 안 나와", "당황스럽네", "참,", "음…", "알았어… 안 돼"
    ],
    "냉소": [
        "잘하고 있겠지", "뻔한 결과", "실망인데", "비웃음", "흥", "관심 없어", "어차피 안 돼", 
        "그게 다야", "하찮아", "하찮군", "별것도 아니지", "뭐 어쩌라고", "그게 다야?"
    ],
    "희망": [
        "잘될 거야", "해낼 수 있어", "희망을 버리지", "포기하지 마", "믿고 있어", "괜찮아질 거야", 
        "다시 시작", "희망은", "희망", "기대해", "할 수 있어", "믿어", "기다려", "기다려보자", 
        "희망을 잃지마", "다시 시작이야", "금방 갈게", "할 수 있어"
    ]
}

# Personality (성격) 키워드 사전: 성격 태그별로 다양한 표현 포함
trait_keywords = {
    "명랑함": [
        "좋아", "놀자", "헤헤", "재밌는", "기뻐", "신나", "웃음", "멋지다", "즐겁다", "신나는", 
        "히히", "축제", "우와", "기쁘", "웃고", "기대돼", "정말 신나", "기분 짱이야", "기대된다", 
        "정말 즐거워", "재밌어 보인다", "만나서 반가워", "멋진 이름이네"
    ],
    "침착함": [
        "침착하게", "흐트러지지", "차분하게", "분석해", "냉정하게", "서두르지", "먼저 생각", 
        "확실히", "계획대로", "진정해", "냉철", "절제", "차분함", "먼저 생각해보자", "계획대로 하자"
    ],
    "충동적": [
        "그냥 질러", "일단 가자", "해봐야", "생각은 나중에", "지금 당장", "뒤는 나중", "무계획", 
        "즉흥", "싫어", "짜증", "그만둬", "화나", "귀찮", "왜 자꾸", "나가", "열받", "뭐야!", 
        "일단 해", "질러!", "멈춰!", "사탕 좀 사고 싶어", "어이!!", "재밌어 보인다"
    ],
    "논리적": [
        "감정은 접어두고", "결과만", "불필요한 감정", "냉정하게", "객관", "논리적", "데이터", 
        "이성적", "논리", "분석", "계산", "확률", "정보", "계획적", "팩트를 기반으로", "데이터에 따르면"
    ],
    "감성적": [
        "찡했어", "외로웠어", "이해하고 싶었어", "눈물이", "감동", "마음이 따뜻", "감정이 북받쳐", 
        "울", "미안", "괜찮아?", "그랬구나", "마음이 아파", "슬펐어", "울컥", "상처받았어", "진심이야", 
        "감동이야", "다정하구나", "고마워", "펜던트", "외롭다"
    ],
    "자신감": [
        "맡겨", "해낼게", "없으면 안 돼", "보여줄게", "자신 있어", "실패하지 않아", "내가 이끌어", 
        "내가 할게", "잘할 수 있어"
    ],
    "소심함": [
        "정말 괜찮을까", "혹시 나 때문에", "튀면 안 돼", "싫어하지 않을까", "가만히 있을게", "소심", 
        "망설여", "부끄러", "작게 말", "머뭇", "소극적", "그럴 리가", "혹시 실수?", "두렵지만", 
        "없단 말이야", "제발", "부탁이야", "안 돼?", "잃어버린"
    ],
    "도도함": [
        "그건 내 취향이 아니야", "기준에 못 미쳐", "설득하려면", "아쉽지만", "더 나은 걸 원해", 
        "관심 없어", "설득할 생각도 없어"
    ],
    "장난기 많음": [
        "장난 아니지", "무서운 척", "당황했지", "놀랐지", "진지한 척 금지", "까불", "웃겨서", 
        "하하", "장난", "놀렸지", "속았지", "웃기지", "장난이야", "재밌지", "까불까불", "진지한 척 하지 마"
    ],
    "비관적": [
        "안 될 거야", "항상 불행", "실패로 끝나겠지", "별 기대 안 해", "희망이 없어", "역시 그럴 줄", "실패야"
    ],
    "현실적": [
        "현실부터", "가능한 것부터", "지금 당장", "실용적인", "결과 중심", "현명하게"
    ],
    "이상주의적": [
        "더 나은 세상", "꿈이 있기에", "가치는 중요해", "이상적", "마음먹기에 달려 있어", 
        "가치 있는 삶", "희망을 이야기해"
    ],
    "사려 깊음": [
        "상처가 되지", "충분히 이해가 돼", "기분을 생각했어", "조심스럽게", "배려", "이해해", 
        "걱정했어", "사려 깊", "상처받을까봐", "조심스럽게 말할게", "고마워", "신사구나", "다정하구나"
    ],
    "통찰력 있음": [
        "진짜 원인은", "두려워하고 있는 거야", "숨겨진 의미", "본질", "간파했어", "문제의 핵심은", 
        "경쟁심이 강하구나"
    ],
    "고집 있음": [
        "난 옳아", "내 방식이야", "고치지 않을 거야", "설득하려 하지 마", "소용없어", "이게 옳아", 
        "끌고 가지 마"
    ],
    "권위적": [
        "내 말이 규칙", "상명하복", "명령이다", "복종해", "위계", "지시를 따라"
    ],
    "의욕적": [
        "시작하자", "성공하겠어", "내가 이끌어줄게", "전진", "목표", "해보자", "가자", "힘내", 
        "이번엔 꼭", "포기하지마", "의욕", "힘내자", "할 수 있어", "금방 갈게"
    ],
    "신비주의적": [
        "언젠가 알게 될 거야", "뜻대로 흘러가", "말해줄 수 없어", "비밀이야", 
        "내가 알려줄 수는 없어", "지금은 밝힐 수 없어"
    ]
}

# 3. 키워드 기반 1차 태깅
def infer_tags_by_keywords(text, existing_traits=None, existing_emotions=None):
    lower_text = text.lower()
    traits_set = set(existing_traits) if existing_traits else set()
    emotions_set = set(existing_emotions) if existing_emotions else set()
    for label, keywords in emotion_keywords.items():
        if any(keyword in lower_text for keyword in keywords):
            emotions_set.add(label)
    for label, keywords in trait_keywords.items():
        if any(keyword in lower_text for keyword in keywords):
            traits_set.add(label)
    return traits_set, emotions_set

# 4. OpenAI SDK v1 방식으로 AI 태깅 함수 구현
from openai import OpenAI
client = OpenAI(api_key="")  # ← 실제 API 키 입력

def infer_tags_via_ai(text):
    allowed_emotions = list(emotion_keywords.keys())
    allowed_traits = list(trait_keywords.keys())
    system_msg = (
        "You are an assistant that labels a given Korean sentence with emotion and personality tags. "
        "Only respond with a JSON object containing two keys: 'emotion' and 'traits'."
    )
    user_msg = (
        f"문장: \"{text}\"\n"
        f"가능한 emotion 태그: {allowed_emotions}\n"
        f"가능한 trait 태그: {allowed_traits}\n"
        "위 문장의 적절한 emotion과 trait를 태그해줘. "
        "반드시 JSON 형식으로만 답변하고, 키는 'emotion', 'traits'이며 값은 태그 목록이 되도록 해줘."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] OpenAI API 호출 오류: {e}")
        return {"emotion": [], "traits": []}

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        try:
            json_start = content.index('{')
            json_end = content.rindex('}')
            json_str = content[json_start:json_end+1]
            result = json.loads(json_str)
        except Exception:
            result = {"emotion": [], "traits": []}

    emotions = result.get("emotion", [])
    traits = result.get("traits", [])
    if isinstance(emotions, str):
        emotions = [emotions]
    if isinstance(traits, str):
        traits = [traits]
    emotions = [e for e in emotions if e in emotion_keywords]
    traits = [t for t in traits if t in trait_keywords]
    return {"emotion": emotions, "traits": traits}

# 5. 대사 데이터 태깅
for char, lines in (data.items() if isinstance(data, dict) else [("ALL", data)]):
    for line in lines:
        text = line.get("text", "")
        existing_traits = line.get("traits", []) or []
        existing_emotions = line.get("emotion", []) or []
        traits_set, emotions_set = infer_tags_by_keywords(text, existing_traits, existing_emotions)
        if len(traits_set) == 0 or len(emotions_set) == 0:
            ai_result = infer_tags_via_ai(text)
            for t in ai_result.get("traits", []):
                traits_set.add(t)
            for e in ai_result.get("emotion", []):
                emotions_set.add(e)
        line["traits"] = list(traits_set)
        line["emotion"] = list(emotions_set)

# 6. 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"태깅 완료: 결과가 '{output_path.name}' 파일로 저장되었습니다.")