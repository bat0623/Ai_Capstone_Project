#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model

# === 0) 설정값 정의 ===
MODEL_NAME    = "/home/remote/Ai_Capstone_Project/SourceCode/polyglot-ko-3.8B"
LORA_DIR      = "/home/remote/Ai_Capstone_Project/AIbigdata_link/Fine-tuning-LoRA"
NPC_JSON_PATH = "/home/remote/Ai_Capstone_Project/secondSourceCode/npc.json"
MAX_LEN       = 512    # prompt+생성 전체 길이 한도
MAX_NEW_TOKENS = 128   # 실제 생성할 토큰 수00

# === 1) 양자화 설정 및 토크나이저 로드 ===
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# === 2) 베이스 모델 로드 및 LoRA 어댑터 적용 ===
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attention.dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.load_adapter(LORA_DIR, adapter_name="default")
model.eval()

# === 3) 텍스트 생성 파이프라인 준비 ===
chat_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    trust_remote_code=True
)

# === 4) NPC JSON 로드 ===
with open(NPC_JSON_PATH, encoding="utf-8") as f:
    npcs = json.load(f)

# === 5) 기본 프롬프트 생성 함수 ===
def generate_base_prompt(npc_info):
    return (
        f"당신은 {npc_info['npc_info_era']}의 {npc_info['npc_info_name']}입니다.\n"
        f"당신의 나이는 {npc_info['npc_info_age']}이며, 성별은 {npc_info['npc_info_gender']}입니다.\n"
        f"당신의 직업은 {npc_info['npc_info_job']}이고, 사회적 지위는 {npc_info['npc_info_socialstatus']}입니다.\n"
        f"플레이어와 당신의 관계는 {npc_info['npc_info_relation']}입니다.\n"
        f"이 세계는 {npc_info['npc_info_era']} 배경을 가집니다.\n"
        "당신은 이 역할에 충실하게 플레이어의 말에 반응합니다."
    )

# === 6) 전체 프롬프트 생성 함수 ===
def generate_full_prompt(npc_info, player_message):
    prompt_base = npc_info.get("npc_prompt")
    if not prompt_base or not prompt_base.strip():
        prompt_base = generate_base_prompt(npc_info)
    # 이름 뒤에 “말씀:”을 붙여 토크나이저가 이름을 인식하도록 강화
    return (
        f"{prompt_base}\n\n"
        f"플레이어: {player_message}\n"
        f"{npc_info['npc_info_name']} 말씀:"
    )

# === 7) NPC 목록 출력 ===
print("=== 사용 가능한 NPC 목록 ===")
for npc in npcs:
    code = npc.get("npc_info_code", npc.get("id", ""))
    name = npc.get("npc_info_name", npc.get("name", ""))
    job  = npc.get("npc_info_job", npc.get("role", ""))
    print(f"{code} - {name} ({job})")
print("\n명령어: '종료' → 현재 NPC 대화 종료, '완전종료' → 프로그램 종료\n")

# === 8) 인터랙티브 대화 루프 ===
program_exit = False
while True:
    sel = input("=== NPC ID 선택: ").strip()
    if sel == "완전종료":
        break
    if sel == "종료":
        continue

    npc = next((n for n in npcs if n.get("npc_info_code") == sel), None)
    if npc is None:
        print("⚠️ 유효하지 않은 ID입니다. 다시 선택해주세요.\n")
        continue

    print(f"\n=== {npc['npc_info_name']} 대화 시작 ===")
    while True:
        user_input = input("플레이어: ").strip()
        if user_input == "완전종료":
            program_exit = True
            break
        if user_input == "종료":
            print(f"=== {npc['npc_info_name']} 대화 종료 ===\n")
            break

        full_prompt = generate_full_prompt(npc, user_input)
        response = chat_pipe(
            full_prompt,
            truncation=True,        # 명시적 토크나이즈 잘림
            max_length=MAX_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        generated = response[0]["generated_text"]
        reply = generated[len(full_prompt):].strip()
        print(f"{npc['npc_info_name']}: {reply}")

    if program_exit:
        break

# === 9) 메모리 및 캐시 정리 ===
del chat_pipe, model, base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\n✅ 프로그램 완전종료 및 메모리 정리 완료")
