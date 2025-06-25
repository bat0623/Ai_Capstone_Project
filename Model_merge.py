import os
import torch
import shutil
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# DeepSpeed 비활성화를 위한 환경 변수 설정
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# 경로 설정
base_model_path = "/home/remote/Ai_Capstone_Project/polyglot-ko-5.8b-chat"
lora_checkpoint_path = "/home/remote/Ai_Capstone_Project/Model3/lora-5.8b-chat copy/checkpoint-189"
merged_model_path = "/home/remote/Ai_Capstone_Project/Model"

def copy_model_configs(base_model_path, merged_model_path):
    """기본 모델의 설정 파일들을 병합된 모델 폴더에 복사합니다."""
    print("📋 모델 설정 파일들 복사 중...")
    
    # 복사할 설정 파일들
    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt"
    ]
    
    copied_count = 0
    for config_file in config_files:
        src_path = os.path.join(base_model_path, config_file)
        dst_path = os.path.join(merged_model_path, config_file)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"  ✅ 복사됨: {config_file}")
                copied_count += 1
            except Exception as e:
                print(f"  ⚠️ 복사 실패: {config_file} - {e}")
        else:
            print(f"  ℹ️ 파일 없음: {config_file} (건너뜀)")
    
    print(f"  📊 총 {copied_count}개 설정 파일이 복사되었습니다.")

def cleanup_training_files(model_path):
    """추론에 불필요한 학습 관련 파일들을 제거합니다."""
    print("6. 학습 관련 파일 정리 중...")
    
    # 제거할 파일 패턴들
    training_files = [
        "optimizer.pt",
        "scheduler.pt", 
        "training_args.bin",
        "trainer_state.json",
        "rng_state.pth",
        "scaler.pt",
        "pytorch_model.bin.index.json",  # sharded 모델의 인덱스 파일
    ]
    
    # 제거할 폴더 패턴들
    training_folders = [
        "checkpoint-*",
        "runs",
        "__pycache__",
    ]
    
    removed_count = 0
    
    # 파일 제거
    for pattern in training_files:
        files = glob.glob(os.path.join(model_path, pattern))
        for file in files:
            try:
                os.remove(file)
                print(f"  ✅ 제거됨: {os.path.basename(file)}")
                removed_count += 1
            except Exception as e:
                print(f"  ⚠️ 제거 실패: {os.path.basename(file)} - {e}")
    
    # 폴더 제거
    for pattern in training_folders:
        folders = glob.glob(os.path.join(model_path, pattern))
        for folder in folders:
            try:
                shutil.rmtree(folder)
                print(f"  ✅ 폴더 제거됨: {os.path.basename(folder)}")
                removed_count += 1
            except Exception as e:
                print(f"  ⚠️ 폴더 제거 실패: {os.path.basename(folder)} - {e}")
    
    # LoRA 관련 파일들 제거 (adapter_* 파일들)
    lora_files = glob.glob(os.path.join(model_path, "adapter_*"))
    for file in lora_files:
        try:
            os.remove(file)
            print(f"  ✅ LoRA 파일 제거됨: {os.path.basename(file)}")
            removed_count += 1
        except Exception as e:
            print(f"  ⚠️ LoRA 파일 제거 실패: {os.path.basename(file)} - {e}")
    
    print(f"  📊 총 {removed_count}개 파일/폴더가 정리되었습니다.")
    
    # 최종 모델에 남은 파일들 확인
    print("\n📁 최종 모델 폴더 내용:")
    for item in sorted(os.listdir(model_path)):
        item_path = os.path.join(model_path, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path) / (1024**2)  # MB
            print(f"  📄 {item} ({size:.1f} MB)")
        else:
            print(f"  📁 {item}/")

print("1. Base 모델 로딩 중...")
# 1. base 모델 로딩 (DeepSpeed 감지 피하려면 device_map=None)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map=None,  # DeepSpeed 회피
    torch_dtype=torch.float16,  # 메모리 효율성
    low_cpu_mem_usage=True
)

print("2. LoRA 어댑터 로딩 및 병합 중...")
# 2. LoRA 어댑터 로딩 및 병합
model = PeftModel.from_pretrained(model, lora_checkpoint_path)
model = model.merge_and_unload()

print("3. 모델을 CPU로 이동 중...")
# 3. CPU로 명시적으로 이동
model = model.to('cpu')

print("4. 병합된 모델 저장 중...")
# 4. 안전한 방법으로 저장 (state_dict 직접 저장)
try:
    # 디렉토리가 없으면 생성
    os.makedirs(merged_model_path, exist_ok=True)
    
    # 모델 설정 파일들 먼저 복사
    copy_model_configs(base_model_path, merged_model_path)
    
    # 모델 저장
    model.save_pretrained(
        merged_model_path,
        safe_serialization=True,  # safetensors 사용
        max_shard_size="5GB"
    )
    print(f"✅ 모델이 성공적으로 저장되었습니다: {merged_model_path}")
    
except Exception as e:
    print(f"❌ 모델 저장 실패: {e}")
    print("대안 방법으로 저장을 시도합니다...")
    
    # 디렉토리가 없으면 생성
    os.makedirs(merged_model_path, exist_ok=True)
    
    # 모델 설정 파일들 먼저 복사
    copy_model_configs(base_model_path, merged_model_path)
    
    # 대안: state_dict로 직접 저장
    torch.save(model.state_dict(), os.path.join(merged_model_path, "pytorch_model.bin"))
    print("✅ state_dict로 모델이 저장되었습니다.")

print("5. 토크나이저 저장 중...")
# 5. tokenizer 저장
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)
print("✅ 토크나이저가 저장되었습니다.")

# 6. 학습 관련 파일 정리
cleanup_training_files(merged_model_path)

print("\n🎉 모델 병합 및 정리가 완료되었습니다!")
print(f"📍 최종 모델 위치: {merged_model_path}")
print("🚀 이제 추론용으로 최적화된 모델을 사용할 수 있습니다!")