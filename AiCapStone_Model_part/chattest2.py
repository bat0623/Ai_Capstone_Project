#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model

# === 0) 설정값 정의 ===
MODEL_DIR     = "/home/remote/Ai_Capstone_Project/model"
NPC_JSON_PATH = "/home/remote/Ai_Capstone_Project/AiCapStone_Model_part/npc.json"
MAX_LEN       = 512    # prompt+생성 전체 길이 한도
MAX_NEW_TOKENS = 128   # 실제 생성할 토큰 수

# === 1) 양자화 설정 및 토크나이저 로드 ===
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# === 2) 베이스 모델 로드 및 LoRA 어댑터 적용 ===
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
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
model.load_adapter(MODEL_DIR, adapter_name="default")
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

# === 5) NPC 목록 출력 ===
print("=== 사용 가능한 NPC 목록 ===")
for npc in npcs:
    code = npc.get("npc_info_code", "")
    name = npc.get("npc_info_name", "")
    job  = npc.get("npc_info_job", "")
    print(f"{code} - {name} ({job})")
print("\n명령어: '종료' → 현재 NPC 대화 종료, '완전종료' → 프로그램 종료\n")

# === 6) 인터랙티브 대화 루프 ===
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

    # 대화 기록 초기화
    conversation_history = []
    print(f"\n=== {npc['npc_info_name']} 대화 시작 ===")
    while True:
        user_input = input("플레이어: ").strip()
        if user_input == "완전종료":
            program_exit = True
            break
        if user_input == "종료":
            print(f"=== {npc['npc_info_name']} 대화 종료 ===\n")
            break

        # 사용자 발화를 기록
        conversation_history.append(f"플레이어: {user_input}")

        # prompt_base 합치기
        prompt_base = npc.get("npc_prompt")
        if isinstance(prompt_base, list):
            prompt_base = "\n".join(prompt_base)
        if not isinstance(prompt_base, str) or not prompt_base.strip():
            raise ValueError("npc_prompt가 설정되어 있지 않습니다.")

        # 전체 프롬프트 생성 (기본 프롬프트 + 대화 기록)
        history_text = "\n".join(conversation_history)
        full_prompt = f"{prompt_base}\n\n{history_text}\n{npc['npc_info_name']} 말씀:"

        # 응답 생성
        response = chat_pipe(
            full_prompt,
            truncation=True,
            max_length=MAX_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        generated = response[0]["generated_text"]
        reply = generated[len(full_prompt):].strip()

        # 모델 응답 기록 및 출력
        conversation_history.append(f"{npc['npc_info_name']}: {reply}")
        print(f"{npc['npc_info_name']}: {reply}")

    if program_exit:
        break

# === 7) 메모리 및 캐시 정리 ===
del chat_pipe, model, base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\n✅ 프로그램 완전종료 및 메모리 정리 완료")
