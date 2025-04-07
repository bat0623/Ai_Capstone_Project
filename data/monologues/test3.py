import json
from pathlib import Path

# 파일 경로 설정 (Windows 절대경로)
input_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_trait_emotion_ready.json")
output_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_trait_emotion_expanded.json")

# 감정 및 성격 추론 함수
def infer_traits_and_emotion_enhanced(text):
    lower = text.lower()
    traits = set()
    emotions = set()

    # 감정 태깅
    if any(word in lower for word in ["우와", "멋져", "즐거워", "해냈어", "좋아", "기대돼", "신나", "행복해", "기뻐", "완전 좋아", "웃겨", "즐겁다"]):
        emotions.add("기쁨")
    if any(word in lower for word in ["외로워", "남지 않았어", "미안해", "잘했어야", "눈물이", "그리워", "슬퍼", "괴로워", "절망", "상실감", "울고 싶어"]):
        emotions.add("슬픔")
    if any(word in lower for word in ["그만해", "무시하는", "끝장이다", "짜증", "화나", "열받아", "못 참아", "분노", "화가", "겁도 없이"]):
        emotions.add("분노")
    if any(word in lower for word in ["어?!", "뭐야", "세상에", "정말 그런 일이", "이럴 수가", "예상 못 했어", "헉", "말도 안돼", "진짜?", "충격이야"]):
        emotions.add("놀람")
    if any(word in lower for word in ["무서워", "나가고 싶어", "다가오지 마", "죽는 거", "도망치고", "겁이 나", "두렵다", "공포", "불길해", "위험해"]):
        emotions.add("두려움")
    if any(word in lower for word in ["괜찮을까", "불안해", "느낌이 안 좋아", "잘못될", "초조해", "긴장돼", "찝찝해", "걱정돼", "불편해", "위험할지도"]):
        emotions.add("불안")
    if any(word in lower for word in ["역겨워", "대기 싫어", "가까이 오지 마", "징그러", "기분 나빠", "혐오스럽다", "불쾌해", "더러워", "냄새나"]):
        emotions.add("혐오")
    if any(word in lower for word in ["어쩌지", "지금 뭐 하지", "예상 못 했는데", "당황했어", "머리가 하얘졌어", "무슨 말", "혼란스러워"]):
        emotions.add("당황")
    if any(word in lower for word in ["잘하고 있겠지", "뻔한 결과", "실망인데", "비웃음", "흥", "관심 없어", "어차피 안 돼", "그게 다야", "하찮아"]):
        emotions.add("냉소")
    if any(word in lower for word in ["잘될 거야", "해낼 수 있어", "희망을 버리지", "포기하지 마", "믿고 있어", "괜찮아질 거야", "다시 시작", "희망은"]):
        emotions.add("희망")

    # 성격 태깅
    if any(word in lower for word in ["좋아", "최고야", "놀자", "헤헤", "재밌는", "기뻐", "신나", "웃음", "멋지다", "즐겁다", "신나는"]):
        traits.add("명랑함")
    if any(word in lower for word in ["침착하게", "흐트러지지", "차분하게", "분석해", "냉정하게", "서두르지", "먼저 생각", "확실히", "계획대로"]):
        traits.add("침착함")
    if any(word in lower for word in ["그냥 질러", "일단 가자", "해봐야", "생각은 나중에", "지금 당장", "뒤는 나중", "무계획", "즉흥"]):
        traits.add("충동적")
    if any(word in lower for word in ["감정은 접어두고", "결과만", "불필요한 감정", "냉정하게", "객관", "논리적", "데이터", "이성적"]):
        traits.add("냉정함")
    if any(word in lower for word in ["찡했어", "외로웠어", "이해하고 싶었어", "눈물이", "감동", "마음이 따뜻", "감정이 북받쳐"]):
        traits.add("감성적")
    if any(word in lower for word in ["맡겨", "해낼게", "없으면 안 돼", "보여줄게", "자신 있어", "실패하지 않아", "내가 이끌어"]):
        traits.add("자신감")
    if any(word in lower for word in ["정말 괜찮을까", "혹시 나 때문에", "미안", "튀면 안 돼", "싫어하지 않을까", "가만히 있을게", "소심"]):
        traits.add("소심함")
    if any(word in lower for word in ["그건 내 취향이 아니야", "기준에 못 미쳐", "설득하려면", "아쉽지만", "더 나은 걸 원해", "관심 없어"]):
        traits.add("도도함")
    if any(word in lower for word in ["장난 아니지", "무서운 척", "당황했지", "놀랐지", "진지한 척 금지", "까불", "웃겨서"]):
        traits.add("장난기 많음")
    if any(word in lower for word in ["안 될 거야", "항상 불행", "실패로 끝나겠지", "별 기대 안 해", "희망이 없어"]):
        traits.add("비관적")
    if any(word in lower for word in ["현실부터", "가능한 것부터", "지금 당장", "실용적인", "결과 중심"]):
        traits.add("현실적")
    if any(word in lower for word in ["더 나은 세상", "꿈이 있기에", "가치는 중요해", "이상적", "마음먹기에 달려 있어"]):
        traits.add("이상주의적")
    if any(word in lower for word in ["논리적으로", "계산해보자", "분석", "팩트", "객관적인", "데이터 중심"]):
        traits.add("논리적")
    if any(word in lower for word in ["상처가 되지", "충분히 이해가 돼", "기분을 생각했어", "조심스럽게", "배려"]):
        traits.add("사려 깊음")
    if any(word in lower for word in ["진짜 원인은", "두려워하고 있는 거야", "숨겨진 의미", "본질", "간파했어"]):
        traits.add("통찰력 있음")
    if any(word in lower for word in ["난 옳아", "내 방식이야", "고치지 않을 거야", "설득하려 하지 마", "소용없어"]):
        traits.add("고집 있음")
    if any(word in lower for word in ["내 말이 규칙", "상명하복", "명령이다", "복종해", "위계"]):
        traits.add("권위적")
    if any(word in lower for word in ["시작하자", "성공하겠어", "내가 이끌어줄게", "전진", "목표"]):
        traits.add("의욕적")
    if any(word in lower for word in ["언젠가 알게 될 거야", "뜻대로 흘러가", "말해줄 수 없어", "비밀이야", "내가 알려줄 수는 없어"]):
        traits.add("신비주의적")

    # Fallback: 아무것도 없으면 중립 처리
    if not traits:
        traits.add("중립적")
    if not emotions:
        emotions.add("")

    return list(traits), list(emotions)

# 데이터 로딩
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 태깅 적용
for character, lines in data.items():
    for line in lines:
        traits, emotions = infer_traits_and_emotion_enhanced(line["text"])
        line["traits"] = traits
        line["emotion"] = emotions

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 저장 완료: {output_path}")
