#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== 1. 설정 ==========
MODEL_PATH = "/home/remote/Ai_Capstone_Project/Model_part2/Merged_model"
JSON_PATH  = "/home/remote/Ai_Capstone_Project/conversation_maker/sample_instruction_info.json"
LOG_PATH   = "chat_history.log"

# ========== 2. 모델 및 토크나이저 로드 ==========
print(f"▶️ 모델 로딩: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception as e:
    print("❌ 모델 로드 실패: 경로를 확인하세요.", file=sys.stderr)
    raise e

# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ========== 3. 캐릭터 데이터 로드 ==========
if not os.path.isfile(JSON_PATH):
    print(f"❌ JSON 파일 없음: {JSON_PATH}", file=sys.stderr)
    sys.exit(1)

print(f"▶️ 캐릭터 데이터 로드: {JSON_PATH}")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

backgrounds_by_code = {bg['code']: bg for bg in data.get('backgrounds', []) if 'code' in bg}
players_by_code     = {pl['code']: pl for pl in data.get('players', [])     if 'code' in pl}
npcs_by_code        = {nc['code']: nc for nc in data.get('npcs', [])        if 'code' in nc}

if not backgrounds_by_code or not players_by_code or not npcs_by_code:
    print("❌ JSON 구조 에러: 'backgrounds','players','npcs' 배열을 확인하세요.", file=sys.stderr)
    sys.exit(1)

print(f"✔️ 로드 완료: 배경 {len(backgrounds_by_code)}개, 플레이어 {len(players_by_code)}명, NPC {len(npcs_by_code)}명")
print("사용 가능한 플레이어 →", ", ".join(f"{c}({d['name']})" for c,d in players_by_code.items()))
print("사용 가능한 NPC     →", ", ".join(f"{c}({d['name']})" for c,d in npcs_by_code.items()))
print("명령어: /p <PlayerCode>, /n <NPCCode>, /exit\n")

# ========== 4. 대화 상태 초기화 ==========
current_player      = None
current_npc         = None
conversation_history = []

# 로그 파일 오픈
log_file = open(LOG_PATH, "w", encoding="utf-8")
print(f"▶️ 대화 로그 저장: {LOG_PATH}\n")

# ========== 5. 프롬프트 생성 함수 ==========
def build_prompt(user_input=None):
    """세계관, 프로필, 대화 이력을 합쳐 모델 입력용 문자열 생성."""
    if current_player is None or current_npc is None:
        return ""

    # (A) 세계관 설명
    bg = backgrounds_by_code.get(current_player.get('background_code', ''), {})
    world_ctx = f"[세계관: {bg.get('era','-')}] {bg.get('description','')}\n\n"

    # (B) 플레이어 프로필
    pl = current_player
    player_ctx = (
        f"[Player: {pl['name']}({pl['code']})]\n"
        f"직업: {pl.get('job','-')} | 신분: {pl.get('social_status','-')} | 성별: {pl.get('gender','-')}\n\n"
    )

    # (C) NPC 프로필
    nc = current_npc
    npc_ctx = (
        f"[NPC: {nc['name']}({nc['code']})]\n"
        f"{nc.get('description','')}\n"
        f"역할: {nc.get('job','-')} | 신분: {nc.get('social_status','-')} | 성별: {nc.get('gender','-')}\n"
        f"관계: {nc.get('relation','-')}\n\n"
    )

    # (D) 대화 이력
    history = ""
    for line in conversation_history:
        history += line + "\n"
    history += "\n"

    # (E) 새 사용자 입력
    if user_input is not None:
        history += f"Player: {user_input}\nNPC: "

    return world_ctx + player_ctx + npc_ctx + "대화:\n" + history

# ========== 6. 메인 채팅 루프 ==========
print("=== 채팅 시작 ===")
while True:
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n(시스템) 채팅 종료.")
        break

    # 종료 명령
    if user_input.lower() in ("/exit","exit","quit"):
        print("(시스템) 종료합니다.")
        break

    # 플레이어 교체 (단축 명령 /p)
    if user_input.startswith("/p "):
        parts = user_input.split(maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ''
        if code in players_by_code:
            current_player = players_by_code[code]
            conversation_history.clear()  # 이력 초기화
            print(f"(시스템) 플레이어 변경 → {code}({current_player['name']})")
        else:
            print("(시스템) 사용법: /p <PlayerCode>")
        continue

    # NPC 교체 (단축 명령 /n)
    if user_input.startswith("/n "):
        parts = user_input.split(maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ''
        if code in npcs_by_code:
            current_npc = npcs_by_code[code]
            print(f"(시스템) NPC 변경 → {code}({current_npc['name']})")
        else:
            print("(시스템) 사용법: /n <NPCCode>")
        continue

    # 캐릭터 미설정 안내
    if current_player is None or current_npc is None:
        print("(시스템) 먼저 /p와 /n 으로 캐릭터를 선택하세요.")
        continue

    # === 대화 처리 ===
    # 1) 이력 및 로그 기록
    conversation_history.append(f"Player: {user_input}")
    log_file.write(f"Player({current_player['code']}-{current_player['name']}): {user_input}\n")

    # 2) 프롬프트 생성
    prompt = build_prompt(user_input=user_input)

    # 3) 토크나이즈 및 device 이동, token_type_ids 제거
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k,v in inputs.items()}

    # 4) 텍스트 생성
    out_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,    # 반복 방지
        no_repeat_ngram_size=3,    # 3-그램 반복 금지
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 5) 답변 디코딩 및 후처리
    raw = tokenizer.decode(
        out_ids[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    ).strip()
    # "Player:" 전 이전, 첫번째 줄만 취함
    reply = raw.split("Player:")[0].split("\n")[0].strip()

    # 6) 출력 및 이력/로그 업데이트
    print(f"NPC({current_npc['name']}): {reply}\n")
    conversation_history.append(f"NPC: {reply}")
    log_file.write(f"NPC({current_npc['code']}-{current_npc['name']}): {reply}\n")
    log_file.flush()

# ========== 7. 메모리 정리 ==========
print("(시스템) 모델 메모리 정리 중...")
del model
del tokenizer
if device.type == 'cuda':
    torch.cuda.empty_cache()
gc.collect()

# 로그 파일 닫기
log_file.close()
