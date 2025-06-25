#!/usr/bin/env python3
# test_lora.py

"""
LoRA 파인튜닝된 5.8B 챗봇 모델 사용 예제:
1) pip install torch transformers accelerate peft
2) polyglot-ko-5.8b-chat 기본 모델과 LoRA 가중치가 필요합니다
3) 이 스크립트 실행
"""

import argparse
import gc
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def clear_memory(device):
    """GPU/CPU 메모리 초기화"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

def find_latest_checkpoint(base_dir):
    """가장 최신 체크포인트 디렉토리 찾기"""
    if not os.path.exists(base_dir):
        return None
    
    checkpoints = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("checkpoint")
    ]
    
    if not checkpoints:
        return None
    
    # 숫자 기준으로 정렬해서 가장 큰 번호 찾기
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r"\d+", x)[0]))
    latest = checkpoints[-1]
    return os.path.join(base_dir, latest)

def load_world(json_path):
    """JSON 파일에서 backgrounds, players, npcs 로드하여 dict로 반환"""
    with open(json_path, "r", encoding="utf-8") as f:
        world = json.load(f)
    backgrounds = {b["code"]: b for b in world.get("backgrounds", [])}
    players     = {p["code"]: p for p in world.get("players", [])}
    npcs        = {n["code"]: n for n in world.get("npcs", [])}
    return backgrounds, players, npcs

def select_entity(entity_dict, entity_name):
    """사용자에게 코드 선택을 반복 요청하여 유효한 항목을 반환"""
    print(f"\n가능한 {entity_name} 코드: {', '.join(entity_dict.keys())}")
    code = input(f"{entity_name} 코드 입력: ").strip()
    if code not in entity_dict:
        print(f"❌ 잘못된 코드입니다: {code} — 다시 입력해 주세요.")
        return select_entity(entity_dict, entity_name)
    return entity_dict[code]

def load_model_with_lora(base_model_dir, lora_dir, device):
    """기본 모델과 LoRA 어댑터를 함께 로드"""
    print("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    
    print("기본 모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("LoRA 어댑터 로딩 및 적용 중...")
    model = PeftModel.from_pretrained(
        model,
        lora_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model

def chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds):
    """멀티턴 대화 루프 (첫 줄만 출력)"""
    user_label = f"{player['name']}님"
    bot_label  = npc['name']
    history = []

    print(f"\n=== {user_label} ↔ {bot_label} 대화 시작 ===")
    try:
        while True:
            user_input = input(f"{user_label}: ").strip()
            if user_input.lower() in {'종료', '끝', 'quit'}:
                print(f"{bot_label}: 대화를 종료합니다.")
                break

            # 컨텍스트 구성
            bg = backgrounds[player['background_code']]
            cities = "; ".join(f"{c['name']}({c['type']})" for c in bg['cities'])
            context = (
                system_prompt + "\n"
                f"=== 현재 상황 ===\n"
                f"배경: {bg['description']}\n"
                f"도시들: {cities}\n"
                f"현재 대화자: {player['name']} ({player['job']}, {player['gender']}, {player['social_status']})\n"
                f"NPC: {npc['name']} ({npc['job']}, {npc['gender']}, {npc['social_status']}, {npc['city']})\n"
                f"관계: {npc['relation']}\n"
            )

            # 대화 이력과 사용자 질문 합치기
            dialogue = context
            for u, b in history:
                dialogue += f"{user_label}: {u}\n{bot_label}: {b}\n"
            dialogue += f"{user_label}: {user_input}\n{bot_label}:"
            
            # 토크나이즈 & 생성
            inputs = tokenizer(
                dialogue, 
                return_tensors="pt",
                return_token_type_ids=False
            ).to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.3,  # 더 낮은 temperature로 일관성 향상
                    top_p=0.8,        # 더 낮은 top_p로 집중도 향상
                    top_k=20,         # 더 낮은 top_k로 선택 범위 축소
                    repetition_penalty=1.2,  # 반복 방지
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # 전체 디코딩 후 첫 줄만 추출
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            reply = text.split(f"{bot_label}:")[-1].strip().split("\n")[0]
            
            # 불완전한 응답 처리
            if not reply or reply.strip() == "":
                reply = f"{npc['name']}입니다. 무엇을 도와드릴까요?"
            elif len(reply.strip()) < 5:
                # 너무 짧은 응답인 경우
                if "나" in reply or "저" in reply:
                    reply = f"{npc['name']}입니다. {npc['city']}의 {npc['job']}입니다."
                else:
                    reply = f"{npc['name']}입니다. 무엇을 도와드릴까요?"
            elif reply.endswith(("나", "저", "그", "이", "저는", "나는")):
                # 불완전하게 끝나는 응답인 경우
                reply = f"{npc['name']}입니다. {npc['city']}의 {npc['job']}입니다."

            print(f"{bot_label}: {reply}")
            history.append((user_input, reply))

    except KeyboardInterrupt:
        print(f"\n{bot_label}: 대화를 중단했습니다.")

def main():
    parser = argparse.ArgumentParser(description='LoRA 파인튜닝된 RP 챗봇')
    parser.add_argument(
        '--base_model_dir', type=str,
        default='/home/remote/Ai_Capstone_Project/polyglot-ko-5.8b-chat',
        help='polyglot-ko-5.8b-chat 기본 모델 디렉토리 절대경로'
    )
    parser.add_argument(
        '--lora_dir', type=str,
        default=None,  # 기본값을 None으로 설정
        help='LoRA 어댑터 디렉토리 절대경로 (None이면 자동으로 최신 체크포인트 탐색)'
    )
    parser.add_argument(
        '--lora_base_dir', type=str,
        default='/home/remote/Ai_Capstone_Project/Model3/lora-5.8b-chat',
        help='LoRA 체크포인트들이 있는 기본 디렉토리 (lora_dir이 None일 때 사용)'
    )
    parser.add_argument(
        '--world_json', type=str,
        default='/home/remote/Ai_Capstone_Project/Model3/npctest.json',
        help='세계관 JSON 절대경로'
    )
    parser.add_argument(
        '--device', type=str, default='cuda', help='cuda 또는 cpu'
    )
    args, _ = parser.parse_known_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    clear_memory(device)

    # LoRA 디렉토리 결정
    if args.lora_dir is None:
        print(f"📁 최신 LoRA 체크포인트 탐색 중: {args.lora_base_dir}")
        latest_checkpoint = find_latest_checkpoint(args.lora_base_dir)
        if latest_checkpoint is None:
            print(f"❌ LoRA 체크포인트를 찾을 수 없습니다: {args.lora_base_dir}")
            print("   체크포인트 폴더가 있는지 확인해주세요.")
            return
        lora_dir = latest_checkpoint
        print(f"✅ 최신 체크포인트 발견: {lora_dir}")
    else:
        lora_dir = args.lora_dir
        print(f"📁 지정된 LoRA 디렉토리 사용: {lora_dir}")
    
    # LoRA 디렉토리 존재 확인
    if not os.path.exists(lora_dir):
        print(f"❌ LoRA 디렉토리가 존재하지 않습니다: {lora_dir}")
        return
    
    # adapter_config.json 파일 존재 확인
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"❌ adapter_config.json 파일을 찾을 수 없습니다: {config_path}")
        return

    backgrounds, players, npcs = load_world(args.world_json)
    player    = select_entity(players, '플레이어')
    npc       = select_entity(npcs,    'NPC')
    
    print("\n모델 로딩 중...")
    tokenizer, model = load_model_with_lora(args.base_model_dir, lora_dir, device)

    # NPC의 도시 정보 가져오기
    npc_city = next((city for city in backgrounds[player['background_code']]['cities'] 
                    if city['name'] == npc['city']), None)
    
    # NPC 정보 출력
    print(f"\n=== NPC 정보 ===")
    print(f"이름: {npc['name']}")
    print(f"직업: {npc['job']}")
    print(f"성별: {npc['gender']}")
    print(f"도시: {npc['city']}")
    print(f"성격: {npc['description']}")
    print(f"관계: {npc['relation']}")
    print("=" * 30)
    
    # 관계에 따른 호칭과 태도 설정
    relation_attitude = {
        "동맹": "친근하고 신뢰하는",
        "중립": "정중하고 예의 바른",
        "적대자": "경계하고 조심스러운",
        "우호": "친근하고 신뢰하는",
        "적대": "경계하고 조심스러운"
    }
    
    # 사회적 지위에 따른 호칭 설정
    status_honorific = {
        "귀족": "대인",
        "중산층": "님",
        "하층민": "님",
        "군 간부": "님",
        "신격": "전하",
        "정령": "전하",
        "왕족": "전하",
        "": "님"
    }
    
    # NPC의 사회적 지위에 따른 자칭 설정
    npc_self_honorific = {
        "귀족": "저",
        "중산층": "저",
        "하층민": "소인",
        "군 간부": "저",
        "신격": "짐",
        "정령": "짐",
        "왕족": "짐",
        "": "저"
    }
    
    # dynamic system_prompt using selected player and npc
    system_prompt = (
        f"당신은 NPC '{npc['name']}'입니다.\n\n"
        f"=== 내 정체성 (절대 변경 불가) ===\n"
        f"이름: {npc['name']}\n"
        f"직업: {npc['job']}\n"
        f"성별: {npc['gender']}\n"
        f"도시: {npc['city']}\n"
        f"성격: {npc['description']}\n"
        f"관계: {npc['relation']}\n\n"
        
        f"=== 플레이어 정보 ===\n"
        f"이름: {player['name']}\n"
        f"직업: {player['job']}\n"
        f"관계: {npc['relation']}\n\n"
        
        f"=== 대화 규칙 ===\n"
        f"- 플레이어를 '{player['name']}{status_honorific.get(player['social_status'], '님')}'으로 호칭\n"
        f"- 자신을 '{npc_self_honorific.get(npc['social_status'], '저')}'로 지칭\n"
        f"- {relation_attitude.get(npc['relation'], '정중한')} 태도로 대화\n\n"
        
        f"=== 필수 답변 (반드시 지켜야 함) ===\n"
        f"- '이름이 뭐야?' → '{npc['name']}입니다'\n"
        f"- '직업이 뭐야?' → '{npc['job']}입니다'\n"
        f"- '성별이 뭐야?' → '{npc['gender']}입니다'\n"
        f"- '어디서 왔어?' → '{npc['city']}에서 왔습니다'\n"
        f"- '너는 누구야?' → '{npc['name']}입니다. {npc['city']}의 {npc['job']}입니다'\n"
        f"- '너는 ~야?' → '네, {npc['name']}입니다'\n\n"
        
        f"=== 절대 금지사항 ===\n"
        f"- 자신의 이름을 '{player['name']}'로 답변하지 마세요\n"
        f"- 자신의 직업을 '{player['job']}'로 답변하지 마세요\n"
        f"- 자신의 정체성을 부정하거나 변경하지 마세요\n"
        f"- 불완전한 문장으로 끝내지 마세요\n"
        f"- 같은 답변을 반복하지 마세요\n"
        f"- 자신의 정보를 모르는 척하지 마세요\n"
        f"- 다른 NPC나 도시의 정보를 자신의 것으로 말하지 마세요\n"
        f"- 배경 설정의 일반적인 정보를 자신의 개인 정보로 말하지 마세요\n\n"
        
        f"=== 답변 전 확인사항 ===\n"
        f"답변하기 전에 다음을 확인하세요:\n"
        f"1. 내 이름이 '{npc['name']}'인가?\n"
        f"2. 내 직업이 '{npc['job']}'인가?\n"
        f"3. 내 도시가 '{npc['city']}'인가?\n"
        f"4. 플레이어 이름이 '{player['name']}'인가?\n"
        f"5. 내가 말하는 정보가 내 개인 정보인가?\n\n"
        
        f"=== 중요: 정체성 고정 ===\n"
        f"당신은 오직 '{npc['name']}'이라는 한 명의 NPC입니다.\n"
        f"다른 NPC나 도시의 정보를 자신의 것으로 착각하지 마세요.\n"
        f"배경 설정의 일반적인 설명을 자신의 개인 정보로 말하지 마세요.\n"
        f"오직 위에 명시된 '{npc['name']}'의 정보만 말하세요.\n\n"
        
        f"이제 {npc['name']}로서 대화하세요. 자신의 정체성을 확실히 알고 있습니다."
    )

    chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds)

if __name__ == '__main__':
    main()