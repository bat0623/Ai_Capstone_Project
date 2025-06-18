#!/usr/bin/env python3

# finetune_lora.py

"""
LoRA 파인튜닝 스크립트 (polyglot-ko-5.8b-chat)
사용법:
1) pip install torch transformers datasets accelerate peft
2) Git LFS로 polyglot-ko-5.8b-chat 클론 및 weights 다운로드
3) train_data.jsonl 준비 ({"instruction":"…","input":"…","output":"…"} 형식)
4) accelerate launch finetune_lora.py
"""
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

import json
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import multiprocessing as mp

# 멀티프로세싱 설정을 main 함수 외부로 이동
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # fork -> spawn으로 변경

def make_prompt(example):
    instr = example["instruction"].strip()
    inp = example["input"].strip()
    tgt = example["output"].strip()
    prompt = f"{instr}\n\n### 사용자 질문:\n{inp}\n\n### 챗봇 답변:"
    return {"prompt": prompt, "target": tgt}

def tokenize_fn(ex, tokenizer):
    full = ex["prompt"] + " " + ex["target"] + tokenizer.eos_token

    tokenized = tokenizer(
        full,
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    input_ids = tokenized["input_ids"]
    prompt_len = len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    tokenized["labels"] = labels
    return tokenized

def main():
    print(f"PyTorch 버전: {torch.__version__}")

    # CPU 코어 수 자동 인식
    cpu_count = os.cpu_count() or 1
    
    # 데이터 전처리용 멀티프로세싱 설정 (보수적으로 절반 사용)
    preprocessing_workers = max(1, cpu_count-1)  # 최대 16개로 제한
    
    # 훈련용 DataLoader 워커는 0으로 유지 (안정성 우선)
    training_workers = 0

    print(f"전체 CPU 코어: {cpu_count}")
    print(f"데이터 전처리 프로세스 수: {preprocessing_workers}")
    print(f"훈련 DataLoader 워커 수: {training_workers}")
    
    # ─── 설정 ──────────────────────────────────────────────────────────────
    MODEL_NAME = "/home/remote/Ai_Capstone_Project/polyglot-ko-3.8b-chat"
    #DATA_PATH = "/home/remote/Ai_Capstone_Project/conversation_1.7G_singleline.jsonl"
    DATA_PATH = "/home/remote/Ai_Capstone_Project/data_100M_singleline.jsonl"
    OUTPUT_DIR = "./lora-3.8b-chat"
    EPOCHS = 3
    LR = 1e-4
    CHUNK_SIZE = 2000  # 한 번에 처리할 데이터 수

    # ─── 토크나이저 & 모델 로드 ────────────────────────────────────────────
    print("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # 패딩 토큰 설정 (중요!)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("기본 모델 로딩 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 양자화된 모델을 LoRA 학습에 맞게 준비
    print("모델을 LoRA 학습에 맞게 준비 중...")
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정 최적화
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense"],
        bias="none",
        modules_to_save=None,
        init_lora_weights=True,
    )

    print("LoRA 모델 설정 중...")
    model = get_peft_model(model, lora_config)
    
    # LoRA 파라미터 활성화 확인
    model.print_trainable_parameters()
    
    # LoRA 파라미터가 제대로 활성화되었는지 확인
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    if trainable_params == 0:
        print("경고: 학습 가능한 파라미터가 없습니다!")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                print(f"LoRA 파라미터 활성화: {name}")
    
    # ─── 데이터셋 로딩 & 전처리 (멀티프로세싱 활용) ─────────────────────────────────
    print("데이터셋 로딩 중...")
    ds = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train",
        cache_dir="./dataset_cache",
        num_proc=preprocessing_workers  # 멀티프로세싱 활용
    )

    total_samples = len(ds)
    print(f"전체 데이터셋 크기: {total_samples}")

    print("프롬프트 생성 중...")
    ds = ds.map(
        make_prompt,
        remove_columns=ds.column_names,
        num_proc=preprocessing_workers,  # 멀티프로세싱 활용
        desc="프롬프트 생성"
    )

    print("토크나이징 중...")
    # tokenizer를 람다 함수로 전달하여 멀티프로세싱에서 사용
    ds = ds.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        remove_columns=["prompt", "target"],
        num_proc=preprocessing_workers,  # 멀티프로세싱 활용
        desc="토크나이징"
    )

    # 전처리 완료 후 메모리 정리
    import gc
    gc.collect()
    print("데이터 전처리 완료 및 메모리 정리")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # ─── 데이터셋을 청크로 나누어 학습 (단일 프로세스로 안정성 확보) ──────────────────────────────────────
    total_iterations = EPOCHS * ((total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE)
    current_iteration = 0
    for epoch in range(EPOCHS):
        print(f"\n에포크 {epoch + 1}/{EPOCHS} 시작")
        for chunk_start in range(0, total_samples, CHUNK_SIZE):
            current_iteration += 1
            progress = current_iteration / total_iterations
            # 진행 바 출력
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            percent = int(progress * 100)
            # ✔️ 줄바꿈 없이 깔끔하게 출력
            print(f"\r전체 진행률: |{bar}| {percent}% ({current_iteration}/{total_iterations})", end="", flush=True)

            chunk_end = min(chunk_start + CHUNK_SIZE, total_samples)
            print()  # 다음 로그 출력을 위해 줄바꿈 확보
            print(f"청크 {chunk_start // CHUNK_SIZE + 1} 처리 중 ({chunk_start}~{chunk_end} /{CHUNK_SIZE})")
            
            chunk_ds = ds.select(range(chunk_start, chunk_end))
            
            # TrainingArguments - 훈련 단계에서는 안정성 우선
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=LR,
                fp16=True,
                logging_steps=50,
                save_steps=100,
                save_total_limit=3,
                max_grad_norm=1.0,
                warmup_ratio=0.1,
                label_names=["labels"],
                gradient_checkpointing=False,
                ddp_find_unused_parameters=False,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                gradient_checkpointing_kwargs={"use_reentrant": False},  # 이 줄 추가
                dataloader_num_workers=training_workers,  # 훈련에서는 0으로 안정성 확보
                optim="adamw_torch",
                report_to="none",
                disable_tqdm=False,
                eval_accumulation_steps=None,
                prediction_loss_only=True,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=chunk_ds,
                data_collator=data_collator,
            )
            
            # 메모리 정리
            torch.cuda.empty_cache()
            print("*** trainer.train() ***")
            
            # 학습 전 파라미터 상태 확인
            has_trainable = any(p.requires_grad for p in model.parameters())
            print(f"학습 가능한 파라미터 존재: {has_trainable}")
            
            try:
                trainer.train()
            except RecursionError as e:
                print(f"재귀 오류 발생: {e}")
                print("모델 래핑 상태를 확인하고 재시도합니다...")
                if hasattr(model, 'module'):
                    model = model.module
                continue
            except Exception as e:
                print(f"학습 중 오류 발생: {e}")
                continue
            
            torch.cuda.empty_cache()
            
            if chunk_end < total_samples:
                print(f"중간 저장 중... (청크 {chunk_start // CHUNK_SIZE + 1})")
                model.save_pretrained(f"{OUTPUT_DIR}/checkpoint_chunk_{chunk_start // CHUNK_SIZE + 1}")
                tokenizer.save_pretrained(f"{OUTPUT_DIR}/checkpoint_chunk_{chunk_start // CHUNK_SIZE + 1}")
    
    # 최종 모델 저장
    print("\n최종 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("학습 완료!")


if __name__ == "__main__":
    main()