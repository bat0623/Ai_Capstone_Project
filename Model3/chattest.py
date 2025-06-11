#!/usr/bin/env python3
# chat.py

"""
5.8B 챗봇 모델 사용 예제:
1) pip install torch transformers accelerate
2) Git LFS로 polyglot-ko-5.8b-chat 클론 및 weights 다운로드
3) 이 스크립트 실행
"""

import argparse
import gc
import json
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

def load_model(model_dir, device):
    """5.8B 챗봇 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device)
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
    parser = argparse.ArgumentParser(description='RP 챗봇 with polyglot-ko-5.8b-chat')
    parser.add_argument(
        '--model_dir', type=str,
        default='/home/remote/Ai_Capstone_Project/polyglot-ko-5.8b-chat',
        help='polyglot-ko-5.8b-chat 디렉토리 절대경로'
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
    tokenizer, model = load_model(args.model_dir, device)

    # dynamic system_prompt using selected player and npc
    system_prompt = (
        f"당신은 중세 판타지 세계관에서 '{player['name']}'님(용사)과 '{npc['name']}'(NPC)의 역할극 대화를 중개하는 AI 챗봇입니다.\n"
        f"- 플레이어는 반드시 '{player['name']}'님으로, NPC는 반드시 '{npc['name']}'으로만 호칭하세요.\n"
        "- 응답은 항상 '속성명: 값' 형태로만 작성합니다. 예) Name: 아룬, Job: 상인\n"
        "- JSON에 없는 정보는 '제공된 데이터에 없습니다.'라고 답변하세요.\n"
    )

    chat_loop(system_prompt, tokenizer, model, device, player, npc, backgrounds)

if __name__ == '__main__':
    main()
