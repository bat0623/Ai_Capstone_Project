import json
from collections import defaultdict
from pathlib import Path

# 원본 파일 경로
file_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger.json")

# 결과 저장 경로
output_path = Path("C:/GitHub/Ai_Capstone_Project/data/monologues/Chrono_Trigger_trait_emotion_ready.json")

# JSON 불러오기
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 캐릭터별 대사 정리 (speaker 기준)
character_dialogues = defaultdict(list)

for interaction in data.get("interactions", []):
    for turn in interaction["turns"]:
        speaker = turn["speaker"]
        text = turn["text"]
        # traits와 emotion 필드를 빈 값으로 추가
        character_dialogues[speaker].append({
            "text": text,
            "traits": [],        # 성격 태그 (비워둠)
            "emotion": ""        # 감정 태그 (비워둠)
        })

# JSON으로 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(character_dialogues, f, ensure_ascii=False, indent=2)

print(f"성공적으로 저장 완료: {output_path}")
