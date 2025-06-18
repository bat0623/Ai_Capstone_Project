import os
import random
import sys
import json
import gc
import torch
import multiprocessing as mp
from datasets import Dataset, disable_caching
from multiprocessing import Queue, Process
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers.safetensors_conversion import auto_conversion
sys.path.append("../conversation_maker")
from auto_conversation_maker import ConversationMaker
import pandas as pd


player_dict = json.load(open("../conversation_maker/sample_instruction_info.json"))['players']
npc_dict = json.load(open("../conversation_maker/npcs_info.json"))
conversation_formated_dict = json.load(open("../conversation_maker/conversation_sample.json"))
conversation_maker = ConversationMaker(player_dict, npc_dict, conversation_formated_dict)

cpu_count = os.cpu_count() or 1
preprocessing_workers = max(1, cpu_count-1)
training_workers = 1

def jsonl_stream_generator():
    try:
        shuffle_player_dict = player_dict.copy()
        random.shuffle(shuffle_player_dict)
        shuffle_npc_dict = npc_dict.copy()
        random.shuffle(shuffle_npc_dict)

        for player in player_dict:
            player_code = player["code"]
            for npc in npc_dict:
                npc_code = npc["code"]
                print(f"New Generator >> player_code: {player_code}\tcode: {npc_code}")
                conv_jsonl_gen = conversation_maker.conversation_generator_jsonl(player_code, npc_code)
                yield from conv_jsonl_gen
    except Exception as e:
        print(e)
        yield None


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
        max_length=256,
        padding="max_length"
    )
    input_ids = tokenized["input_ids"]
    prompt_len = len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    tokenized["labels"] = labels
    return tokenized

disable_caching()
def preprocess_chunk(raw_chunk, tokenizer):
    ds = Dataset.from_list(raw_chunk)
    print("dataset sample:",ds[:5])
    print("MAP for make_prompt")
    ds = ds.map(make_prompt, remove_columns=ds.column_names,)
    print("MAP for tokenize_fn")
    ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer), remove_columns=["prompt", "target"],)
    return ds


def chunk_producer(queue: Queue, tokenizer, chunk_size):
    gen = jsonl_stream_generator()
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                gen_data=next(gen)
                chunk.append(gen_data)
        except StopIteration:
            if chunk:
                print("StopIteration")
                queue.put(preprocess_chunk(chunk, tokenizer))
            break
        except Exception as e:
            print(f"[제너레이터] 오류 발생: {e}")
            continue
        print(f"put_chunk: {chunk[-1]}")
        queue.put(preprocess_chunk(chunk, tokenizer))


def get_next_chunk(queue: Queue):
    data=queue.get()
    print("get_next_chunk: ")
    print(f"queue.qsize: {queue.qsize()}")
    return data


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"PyTorch 버전: {torch.__version__}")
    print(f"전체 CPU 코어: {cpu_count}")
    MODEL_NAME = "/home/remote/Ai_Capstone_Project/polyglot-ko-5.8b-chat"
    OUTPUT_DIR = "./lora-5.8b-chat"
    EPOCHS = 3
    LR = 1e-4
    CHUNK_SIZE = 12000
    CHUNKS_PER_EPOCH = 100

    print("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
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

    print("모델을 LoRA 학습에 맞게 준비 중...")
    model = prepare_model_for_kbit_training(model)
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
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=True,
        num_train_epochs=1,
        learning_rate=LR,
        fp16=True,
        logging_steps=100,
        save_steps=1,
        save_total_limit=3,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        label_names=["labels"],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=training_workers,
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=False,
        eval_accumulation_steps=None,
        prediction_loss_only=True,
    )

    queue = Queue(maxsize=1)
    producer = Process(target=chunk_producer, args=(queue, tokenizer, CHUNK_SIZE))
    producer.start()

    current_iteration = 0
    total_iterations = EPOCHS * CHUNKS_PER_EPOCH

    for epoch in range(EPOCHS):
        print(f"\n에포크 {epoch + 1}/{EPOCHS} 시작")
        for chunk_id in range(CHUNKS_PER_EPOCH):
            current_iteration += 1
            progress = current_iteration / total_iterations
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            percent = int(progress * 100)
            print(f"\r전체 진행률: |{bar}| {percent}% ({current_iteration}/{total_iterations})", end="", flush=True)

            print(f"\n청크 {chunk_id + 1} 처리 중...")
            chunk_ds = get_next_chunk(queue)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=chunk_ds,
                data_collator=data_collator,
            )

            torch.cuda.empty_cache()
            print("*** trainer.train() ***")
            try:
                trainer.train()
            except RecursionError as e:
                print(f"재귀 오류 발생: {e}")
                if hasattr(model, 'module'):
                    model = model.module
                continue
            except Exception as e:
                print(f"학습 중 오류 발생: {e}")
                continue

            torch.cuda.empty_cache()

            print(f"중간 저장 중... (청크 {chunk_id + 1})")
            model.save_pretrained(f"{OUTPUT_DIR}/checkpoint_chunk_{chunk_id + 1}")
            tokenizer.save_pretrained(f"{OUTPUT_DIR}/checkpoint_chunk_{chunk_id + 1}")

    print("\n최종 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("학습 완료!")
