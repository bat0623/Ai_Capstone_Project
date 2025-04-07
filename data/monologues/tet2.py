import json
from pathlib import Path

# 파일 경로 설정 (Windows 절대경로)
input_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_trait_emotion_ready.json")
output_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_trait_emotion_expanded.json")

# 감정 및 성격 추론 함수 (세부화된 키워드 대응)
def infer_traits_and_emotion_enhanced(text):
    lower = text.lower()
    traits = set()
    emotions = set()

    # 감정 태깅
    if any(word in lower for word in ["좋아", "기뻐", "즐거워", "행복", "멋져", "신나", "반가워", "다행", "기대돼", "재밌어", "정말 좋아"]):
        emotions.add("기쁨")
    if any(word in lower for word in ["화가", "짜증", "열받", "왜 그래", "꺼져", "짜증나", "너 때문", "도저히", "진짜", "이럴 수가"]):
        emotions.add("분노")
    if any(word in lower for word in ["슬퍼", "외로워", "우울", "눈물", "미안", "그리워", "비참", "포기", "죽고", "아파", "괴로워"]):
        emotions.add("슬픔")
    if any(word in lower for word in ["무서워", "두려워", "불안", "떨려", "망설여", "겁", "긴장", "불편", "망했다"]):
        emotions.add("두려움")
    if any(word in lower for word in ["놀라", "진짜?", "뭐라고", "정말?", "어떻게", "대박", "헐", "세상에", "이럴 수가"]):
        emotions.add("놀람")
    if any(word in lower for word in ["희망", "괜찮아", "할 수 있어", "믿어", "기다려", "포기하지 마", "기대해"]):
        emotions.add("희망")
    if any(word in lower for word in ["혐오", "역겨워", "불쾌", "짜증나", "징그러", "꺼져", "지긋지긋", "더러워"]):
        emotions.add("혐오")

    # 성격 태깅
    if any(word in lower for word in ["히히", "좋아", "재밌", "축제", "신나", "기대돼", "놀자", "헤헤", "우와", "기쁘", "웃고"]):
        traits.add("명랑함")
    if any(word in lower for word in ["싫어", "짜증", "그만둬", "화나", "귀찮", "왜 자꾸", "짜증나", "나가", "열받"]):
        traits.add("충동적")
    if any(word in lower for word in ["논리", "분석", "이성적", "계산", "확률", "객관", "정보", "데이터", "계획적"]):
        traits.add("논리적")
    if any(word in lower for word in ["조심", "망설여", "불안", "부끄러", "작게 말", "소심", "머뭇", "소극적"]):
        traits.add("소심함")
    if any(word in lower for word in ["울", "눈물", "미안", "괜찮아?", "그랬구나", "마음이 아파", "슬펐어"]):
        traits.add("감성적")
    if any(word in lower for word in ["해보자", "가자", "시작하자", "힘내", "이번엔 꼭", "포기하지마", "전진", "의욕"]):
        traits.add("의욕적")
    if any(word in lower for word in ["냉정하게", "침착하게", "차분히", "진정해", "먼저 생각해", "냉철", "절제"]):
        traits.add("침착함")
    if any(word in lower for word in ["하하", "장난", "놀렸지", "속았지", "웃기지", "장난이야", "까불", "재밌지"]):
        traits.add("장난기 많음")
    if any(word in lower for word in ["이해해", "배려", "미안해", "생각했어", "사려 깊", "걱정했어"]):
        traits.add("사려 깊음")
    if any(word in lower for word in ["네가 누구", "자기소개", "만나서 반가워", "처음 뵙겠습니다", "나는 누구", "너는", "소개할게"]):
        traits.add("차분함")

    # Fallback
    if not traits:
        traits.add("")
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