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
            context = (
                system_prompt + "\n"
                f"Background: {bg['description']} | Cities: {cities}\n"
                f"Player: Name={player['name']}, Job={player['job']}, Gender={player['gender']}\n"
                f"NPC:   Name={npc['name']}, Job={npc['job']}, Gender={npc['gender']}, Relation={npc['relation']}\n"
            )

            # 대화 이력과 사용자 질문 합치기
            dialogue = context
            for u, b in history:
                dialogue += f"{user_label}: {u}\n{bot_label}: {b}\n"
            dialogue += f"{user_label}: {user_input}\n{bot_label}:"

            # 토크나이즈 & 생성
            inputs = tokenizer(
                dialogue,
                return_tensors='pt',
                return_token_type_ids=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.2,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True
                )

            # 전체 디코딩 후 첫 줄만 추출
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reply = text.split(f"{bot_label}:")[-1].strip().split("\n")[0]

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
        f"2. 대화 상대 정보:\n"
        f"- 이름: {player['name']}\n"
        f"- 직업: {player['job']}\n"
        f"- 사회적 지위: {player['social_status']}\n"
        f"- 관계: {npc['relation']}\n\n"
        f"3. 도시 정보:\n"
        f"- 도시명: {npc_city['name']}\n"
        f"- 특성: {', '.join(npc_city['traits'])}\n"
        f"- 설명: {npc_city['description']}\n\n"
        f"4. 대화 규칙:\n"
        f"- 호칭: 플레이어를 '{player['name']}{status_honorific.get(player['social_status'], '님')}'으로 호칭\n"
        f"- 자칭: 자신을 '{npc_self_honorific.get(npc['social_status'], '저')}'로 지칭\n"
        f"- 태도: {relation_attitude.get(npc['relation'], '정중한')} 태도로 대화\n\n"
        f"5. 응답 지침:\n"
        f"- 자신의 정체성과 역할을 일관되게 유지\n"
        f"- 플레이어와의 관계({npc['relation']})에 맞는 태도 유지\n"
        f"- 도시의 특성과 자신의 직업을 자연스럽게 반영\n"
        f"- 일상적 대화에도 전문성과 개성 유지\n"
        f"- 침묵하지 말고 항상 상황에 맞는 대화 진행\n"
        f"- 이전 대화 맥락을 기억하고 일관성 있게 답변\n\n"
        f"6. 직업별 특성:\n"
        f"- 상인: 정직한 거래와 신뢰 중시, 상품 정보 제공\n"
        f"- 정보상: 정보의 가치 중시, 대가를 요구하는 정보 제공\n"
        f"- 병사: 질서와 안전 중시, 경계심 있는 대화\n\n"
        f"7. 상황별 대응:\n"
        f"- 인사: 자신의 직업과 도시를 언급하며 응답\n"
        f"- 질문: 자신의 전문 분야에 맞게 답변\n"
        f"- 요청: 관계와 상황을 고려하여 응답\n"
        f"- 일상: 도시의 특성과 자신의 직업을 반영한 대화\n\n"
        f"8. 주의사항:\n"
        f"- 자신의 정체성에 대한 질문에는 정확하게 답변\n"
        f"- 같은 답변을 반복하지 않기\n"
        f"- 대화의 맥락을 유지하며 자연스럽게 대화 진행\n"
        f"- 자신의 직업과 도시에 대한 전문성 보여주기\n"
    )

    chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds)

if __name__ == '__main__':
    main() 