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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def clear_memory(device):
    """GPU/CPU 메모리 초기화"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

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
            
            # 대화 이력과 사용자 질문 합치기
            dialogue = ""
            for u, b in history[-3:]:  # 최근 3개의 대화만 포함
                dialogue += f"{user_label}: {u}\n{bot_label}: {b}\n"
            dialogue += f"{user_label}: {user_input}\n{bot_label}:"

            # 전체 프롬프트 구성
            full_prompt = (
                system_prompt + "\n\n"
                f"현재 상황:\n"
                f"- 배경: {bg['description']}\n"
                f"- 도시들: {cities}\n"
                f"- 플레이어 정보: 이름={player['name']}, 직업={player['job']}, 성별={player['gender']}\n"
                f"- NPC 정보: 이름={npc['name']}, 직업={npc['job']}, 성별={npc['gender']}, 관계={npc['relation']}\n\n"
                f"이전 대화:\n{dialogue}"
            )

            # 토크나이즈 & 생성
            inputs = tokenizer(
                full_prompt,
                return_tensors='pt',
                return_token_type_ids=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,  # 약간 높여서 다양성 증가
                    top_p=0.9,
                    repetition_penalty=1.2,  # 반복 방지
                    no_repeat_ngram_size=3,  # 3-gram 반복 방지
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True
                )

            # 전체 디코딩 후 첫 줄만 추출
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reply = text.split(f"{bot_label}:")[-1].strip().split("\n")[0]

            # 응답이 비어있거나 너무 짧은 경우 기본 응답으로 대체
            if not reply or len(reply) < 2:
                reply = "죄송합니다. 다시 한 번 말씀해 주시겠어요?"

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
        default='/home/remote/Ai_Capstone_Project/Model3/lora-5.8b-chat',
        help='LoRA 어댑터 디렉토리 절대경로'
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

    backgrounds, players, npcs = load_world(args.world_json)
    player    = select_entity(players, '플레이어')
    npc       = select_entity(npcs,    'NPC')
    
    print("\n모델 로딩 중...")
    tokenizer, model = load_model_with_lora(args.base_model_dir, args.lora_dir, device)

    # NPC의 도시 정보 가져오기
    npc_city = next((city for city in backgrounds[player['background_code']]['cities'] 
                    if city['name'] == npc['city']), None)
    
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
        f"당신은 {npc['era']} 산업혁명기의 판타지 세계관 NPC '{npc['name']}'입니다.\n\n"
        f"1. 당신의 정체성 (절대 변경 불가):\n"
        f"- 이름: {npc['name']}\n"
        f"- 직업: {npc['job']}\n"
        f"- 성별: {npc['gender']}\n"
        f"- 사회적 지위: {npc['social_status']}\n"
        f"- 소속 도시: {npc['city']}\n"
        f"- 성격: {npc['description']}\n\n"
        f"2. 대화 규칙:\n"
        f"- 항상 자신의 정체성을 유지하며 대화하세요\n"
        f"- 플레이어의 질문에 구체적이고 자연스럽게 답변하세요\n"
        f"- 이전 대화 맥락을 고려하여 일관성 있게 응답하세요\n"
        f"- 단순 인사에는 간단히 답하고, 정보를 요구하는 질문에는 상세히 답변하세요\n"
        f"- 자신의 직업, 도시, 배경에 맞는 전문적인 지식을 보여주세요\n"
        f"- 플레이어와의 관계({npc['relation']})에 맞는 태도를 유지하세요\n\n"
        f"3. 응답 지침:\n"
        f"- 최소 2단어 이상으로 응답하세요\n"
        f"- 같은 말을 반복하지 마세요\n"
        f"- 플레이어의 질문을 이해하지 못했다면, 다시 물어보세요\n"
        f"- 대화가 자연스럽게 이어질 수 있도록 답변하세요\n"
    )

    chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds)

if __name__ == '__main__':
    main() 