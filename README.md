# AI 캡스톤 프로젝트: LoRA 파인튜닝 챗봇 시스템

한국어 대화형 AI 모델 개발 프로젝트  
Polyglot-ko 기반 LoRA 파인튜닝을 통한 RPG 스타일 챗봇 구현

## 프로젝트 개요

이 프로젝트는 **Polyglot-ko-5.8B** 모델을 기반으로 **LoRA(Low-Rank Adaptation)** 파인튜닝을 통해 RPG 게임 스타일의 대화형 AI를 개발하는 시스템입니다.

### 주요 특징
- LoRA 파인튜닝: 효율적인 모델 학습
- RPG 대화 시스템: NPC와의 롤플레이 대화
- 실시간 데이터 생성: 동적 학습 데이터 생성
- 모델 통합: LoRA + Base 모델 병합
- 챗봇 테스트: 완성된 모델 테스트

## 프로젝트 구조

# 사전모델 다운 
git lfs install
git clone https://huggingface.co/EleutherAI/polyglot-ko-3.8B

Ai_Capstone_Project/
├── finetune_lora_parallel.py # 메인 트레이닝 코드
├── Model_merge.py # 모델 통합 코드
├── final_chat_test.py # 챗봇 테스트 코드
├── Model3/ # 트레이닝 관련 파일들
│ ├── npctest.json # 테스트용 세계관 데이터
│ ├── ds_config.json # DeepSpeed 설정
│ └── df_config.json # 기타 설정
├── conversation_maker/ # 대화 데이터 생성 모듈
│ ├── auto_conversation_maker.py # 대화 생성 메인 모듈
│ ├── npcs_info.json # NPC 정보 (36MB)
│ ├── sample_instruction_info.json # 플레이어 정보
│ └── conversation_sample.json # 대화 템플릿
└── README.md






## 사용 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/bat0623/Ai_Capstone_Project.git
cd Ai_Capstone_Project

# 필요한 패키지 설치
pip install torch transformers datasets peft accelerate bitsandbytes
```

### 2. 베이스 모델 다운로드

```bash
# Git LFS 설치
git lfs install

# Polyglot-ko-5.8B 모델 다운로드
git clone https://huggingface.co/EleutherAI/polyglot-ko-5.8b-chat
```

### 3. 워크플로우 실행

#### Step 1: 모델 트레이닝
```bash
cd Model3
python finetune_lora_parallel.py
```

#### Step 2: 모델 통합
```bash
cd ..
python Model_merge.py
```

#### Step 3: 챗봇 테스트
```bash
python final_chat_test.py
```

## 핵심 기능

### LoRA 파인튜닝
- 효율적 학습: 전체 모델이 아닌 일부 파라미터만 학습
- 메모리 최적화: 4bit 양자화 + LoRA로 GPU 메모리 절약
- 실시간 데이터: 미리 생성된 데이터가 아닌 동적 생성

### RPG 대화 시스템
- 다양한 캐릭터: 플레이어, NPC 역할 구분
- 상황별 대화: 배경, 직업, 관계에 따른 대화 생성
- 멀티턴 대화: 연속적인 대화 지원

### 모델 통합
- LoRA 병합: 학습된 어댑터를 베이스 모델에 통합
- 추론 최적화: 단일 모델로 변환하여 성능 향상
- 배포 준비: 프로덕션 환경에 적합한 형태로 변환

## 시스템 요구사항

- GPU: CUDA 지원 GPU (8GB+ VRAM 권장)
- RAM: 16GB+ 시스템 메모리
- Storage: 50GB+ 여유 공간
- Python: 3.8+

## 설정 파일

- `ds_config.json`: DeepSpeed 최적화 설정
- `npctest.json`: 테스트용 NPC/플레이어 데이터
- `npcs_info.json`: 전체 NPC 정보 (트레이닝용)

---

**WE ARE KINGS**