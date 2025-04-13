import os
import sys
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ✅ Windows 환경일 때 CUDA DLL 경로 추가
if platform.system() == "Windows":
    dll_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin"
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)

# 모델 경로
model_path = "./polyglot-ko-3.8B"

# 8bit quantization 구성
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

print("🔄 모델 및 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
except Exception as e:
    print("❌ 모델 로딩 실패:", e)
    print("✅ float32 fallback으로 재시도 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )

# 프롬프트 예시
prompt = "Q: 세상에서 가장 맛있는 음식은 뭐야?\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
inputs.pop("token_type_ids", None)

print("🧠 텍스트 생성 중...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📢 생성된 텍스트:\n")
print(result)
