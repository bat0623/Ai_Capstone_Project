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

import json

from datasets import load_dataset

import torch

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM,

    Trainer,

    TrainingArguments,

    DataCollatorForSeq2Seq

)

from peft import LoraConfig, get_peft_model, TaskType
# tokenizer 병렬처리 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# CUDA 메모리 할당 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# CPU 코어 수 자동 인식 (맵핑·DataLoader 워커)
cpu_count = os.cpu_count() or 1
n_proc = max(1, cpu_count - 1)
print(f"전체 CPU 코어: {cpu_count}, 데이터 맵핑 프로세스 수: {n_proc}")

# ─── 설정 ──────────────────────────────────────────────────────────────
MODEL_NAME = "/home/remote/Ai_Capstone_Project/polyglot-ko-5.8b-chat"
DATA_PATH = "/home/remote/Ai_Capstone_Project/data_singleline.jsonl"
OUTPUT_DIR = "./lora-5.8b-chat"
BATCH_SIZE = 1
# EPOCHS     = 5
EPOCHS = 1
LR = 1e-4
CHUNK_SIZE = 70  # 한 번에 처리할 데이터 수

# ─── 토크나이저 & 모델 로드 ────────────────────────────────────────────
print("토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
print("기본 모델 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# LoRA 설정 최적화
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["query_key_value", "dense"],
    bias="none",
    modules_to_save=None,
    init_lora_weights=True,
)

print("LoRA 모델 설정 중...")
model = get_peft_model(model, lora_config)
# 모델의 모든 파라미터에 대해 requires_grad 설정
for param in model.parameters():
    param.requires_grad = True

model.print_trainable_parameters()  # 학습 가능한 파라미터 수 출력
# ─── 데이터셋 로딩 & 전처리 (한 번만) ─────────────────────────────────
print("데이터셋 로딩 중...")
ds = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
    cache_dir="./dataset_cache",
    num_proc=n_proc
)

total_samples = len(ds)
print(f"전체 데이터셋 크기: {total_samples}")


def make_prompt(example):
    instr = example["instruction"].strip()
    inp = example["input"].strip()
    tgt = example["output"].strip()
    prompt = f"{instr}\n\n### 사용자 질문:\n{inp}\n\n### 챗봇 답변:"
    return {"prompt": prompt, "target": tgt}


def tokenize_fn(ex):
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


print("프롬프트 생성 중...")
ds = ds.map(
    make_prompt,
    remove_columns=ds.column_names,
    num_proc=n_proc
)

print("토크나이징 중...")
ds = ds.map(
    tokenize_fn,
    remove_columns=["prompt", "target"],
    num_proc=n_proc
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)
# ─── 데이터셋을 청크로 나누어 학습 ──────────────────────────────────────
for epoch in range(EPOCHS):
    print(f"\n에포크 {epoch + 1}/{EPOCHS} 시작")
    for chunk_start in range(0, total_samples, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_samples)
        print(f"\n청크 {chunk_start // CHUNK_SIZE + 1} 처리 중 ({chunk_start}~{chunk_end})")
        # 이미 전처리된 ds에서 슬라이스만 수행
        chunk_ds = ds.select(range(chunk_start, chunk_end))
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=128,
            num_train_epochs=1,
            learning_rate=LR,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            optim="adamw_torch",
            optim_args="offload_optimizer=True",
            max_grad_norm=0.3,
            warmup_ratio=0.05,
            label_names=["labels"],
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=n_proc
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=chunk_ds,
            data_collator=data_collator,
        )
        trainer.train()
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
