import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from build_dataset_from_folders import build_dataset_from_folders

def main():
    model_name = "beomi/polyglot-ko-3.8b"
    print(f"Loading base model '{model_name}' with float16 precision...")

    # ✅ 8bit 제거, float16으로 로딩
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # ✅ LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    # ✅ 데이터셋 로드
    data_path = r"C:\github\Ai_Capstone_Project\data"
    print(f"Building dataset from '{data_path}'...")
    dataset = build_dataset_from_folders(data_path)

    # ✅ 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "right"

    # ✅ 전처리 함수
    def tokenize_example(example):
        prompt = example["prompt"]
        response = example["response"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + response_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_example, remove_columns=["prompt", "response"])

    # ✅ 동적 패딩
    def data_collator(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_masks, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_masks.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    # ✅ 학습 파라미터
    output_dir = "./finetuned-polyglot"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        logging_steps=50,
        save_strategy="epoch",
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=True,
        optim="adamw_torch",
        report_to="none"
    )

    # ✅ Trainer 초기화
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )

    # ✅ 학습 시작
    print("Starting fine-tuning...")
    trainer.train()

    # ✅ LoRA 어댑터 저장
    trainer.model.save_pretrained(output_dir)
    print(f"✅ LoRA fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()
