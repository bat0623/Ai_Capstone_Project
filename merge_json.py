import os
import json
from glob import glob
from eunjeon import Mecab

# 데이터셋 폴더 경로
dataset_folder = "./dataset"
output_json = "merged_dataset.json"

# 형태소 분석기 초기화
mecab = Mecab()

# 병합될 JSON 데이터 초기화
merged_data = {"dataset": []}

# 현재 id (고유번호)를 위한 카운터 초기화
current_id = 1

# dataset 폴더 내의 모든 JSON 파일 가져오기
json_files = glob(os.path.join(dataset_folder, "*.json"))

print(f"총 발견된 JSON 파일 수: {len(json_files)}")

# JSON 파일을 하나씩 로드하여 처리
for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        json_content = json.load(f)

        # 대화 텍스트 추출
        conversations = json_content['info'][0]['annotations']['text']
        lines = conversations.strip().split("\n")

        # 한 줄씩 전처리 및 형태소 분석
        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue

            speaker_label, text = line.split(":", 1)
            speaker_label = speaker_label.strip()
            text = text.strip()

            # 형태소 분석 (토큰화)
            tokens = mecab.morphs(text)

            # JSON entry 생성
            json_entry = {
                "id": current_id,
                "text": text,
                "tokens": tokens,
                "metadata": {
                    "speaker": f"화자_{speaker_label}",
                    "source_file": os.path.basename(json_file),
                    "category": "일상대화"
                }
            }

            # 데이터 추가
            merged_data["dataset"].append(json_entry)
            current_id += 1

# 병합된 데이터를 최종 저장
with open(output_json, "w", encoding="utf-8") as output_file:
    json.dump(merged_data, output_file, ensure_ascii=False, indent=2)

print(f"모든 데이터 병합 완료! 최종 데이터 개수: {len(merged_data['dataset'])}")
print(f"병합된 데이터가 '{output_json}'에 저장되었습니다.")
