#!/usr/bin/env python3
# final_chat_test.py

"""
병합된 최종 모델 채팅 테스트 스크립트:
- 병합된 모델을 단일 모델로 로드
- NPC와의 롤플레이 대화 테스트
- 최종 배포용 버전
"""

import argparse
import gc
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def load_merged_model(model_dir, device):
    """병합된 최종 모델을 로드"""
    print("🚀 최종 모델 로딩 중...")
    print(f"📁 모델 경로: {model_dir}")
    
    # 토크나이저 로드
    print("  📝 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    
    # 병합된 모델 로드
    print("  🧠 모델 로딩...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("✅ 최종 모델 로딩 완료!")
    return tokenizer, model

def generate_response(dialogue, tokenizer, model, device, max_new_tokens=100):
    """단일 응답 생성"""
    inputs = tokenizer(
        dialogue,
        return_tensors='pt',
        return_token_type_ids=False,
        truncation=True,
        max_length=1024  # 컨텍스트 길이 제한
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )
    
    # 전체 디코딩 후 첫 줄만 추출
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds):
    """멀티턴 대화 루프"""
    user_label = f"{player['name']}님"
    bot_label  = npc['name']
    history = []

    print(f"\n=== {user_label} ↔ {bot_label} 대화 시작 ===")
    print("💡 '종료', '끝', 'quit' 입력하면 대화를 종료합니다.\n")
    
    try:
        while True:
            user_input = input(f"{user_label}: ").strip()
            if user_input.lower() in {'종료', '끝', 'quit'}:
                print(f"{bot_label}: 대화를 종료합니다. 즐거운 시간이었습니다!")
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

            # 대화 이력과 사용자 질문 합치기 (최근 5턴만 유지)
            dialogue = context
            recent_history = history[-5:] if len(history) > 5 else history
            for u, b in recent_history:
                dialogue += f"{user_label}: {u}\n{bot_label}: {b}\n"
            dialogue += f"{user_label}: {user_input}\n{bot_label}:"

            # 응답 생성
            print(f"{bot_label}: ", end="", flush=True)
            full_text = generate_response(dialogue, tokenizer, model, device)
            reply = full_text.split(f"{bot_label}:")[-1].strip().split("\n")[0]

            print(reply)
            history.append((user_input, reply))

    except KeyboardInterrupt:
        print(f"\n{bot_label}: 대화를 중단했습니다.")

def main():
    parser = argparse.ArgumentParser(description='병합된 최종 모델 채팅 테스트')
    parser.add_argument(
        '--model_dir', type=str,
        default='/home/remote/Ai_Capstone_Project/Model',
        help='병합된 모델 디렉토리 절대경로'
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

    # 모델 디렉토리 존재 확인
    if not os.path.exists(args.model_dir):
        print(f"❌ 모델 디렉토리가 존재하지 않습니다: {args.model_dir}")
        print("먼저 Model_merge.py를 실행하여 모델을 병합해주세요.")
        return
    
    # 세계관 데이터 로드
    try:
        backgrounds, players, npcs = load_world(args.world_json)
    except FileNotFoundError:
        print(f"❌ 세계관 JSON 파일을 찾을 수 없습니다: {args.world_json}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파일 파싱 오류: {e}")
        return
    
    # 플레이어와 NPC 선택
    player = select_entity(players, '플레이어')
    npc = select_entity(npcs, 'NPC')
    
    # 병합된 모델 로드
    try:
        tokenizer, model = load_merged_model(args.model_dir, device)
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return

    # NPC의 도시 정보 가져오기
    npc_city = next((city for city in backgrounds[player['background_code']]['cities'] 
                    if city['name'] == npc['city']), None)
    
    if not npc_city:
        print(f"⚠️ NPC의 도시 정보를 찾을 수 없습니다: {npc['city']}")
        npc_city = {'name': npc['city'], 'traits': ['일반'], 'description': '평범한 도시'}
    
    # 관계에 따른 호칭과 태도 설정
    relation_attitude = {
        "동맹": "친근하고 신뢰하는",
        "중립": "정중하고 예의 바른",
        "적대자": "경계하고 조심스러운",
        "우호": "친근하고 신뢰하는",
        "적대": "경계하고 조심스러운"
    }
    
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
    
    # 시스템 프롬프트 생성
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
        f"- 이전 대화 맥락을 기억하고 일관성 있게 답변\n"
        f"- 간결하고 자연스러운 대화 유지\n\n"
        f"6. 주의사항:\n"
        f"- 자신의 정체성에 대한 질문에는 정확하게 답변\n"
        f"- 같은 답변을 반복하지 않기\n"
        f"- 대화의 맥락을 유지하며 자연스럽게 대화 진행\n"
        f"- 자신의 직업과 도시에 대한 전문성 보여주기\n"
    )

    print(f"\n🎯 최종 모델 채팅 테스트 시작!")
    print(f"📁 모델: {args.model_dir}")
    print(f"🎭 플레이어: {player['name']} ({player['job']})")
    print(f"🤖 NPC: {npc['name']} ({npc['job']}, {npc['city']})")
    print(f"🤝 관계: {npc['relation']}")
    
    chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds)

if __name__ == '__main__':
    main() 