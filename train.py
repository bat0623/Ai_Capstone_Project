import os
import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import load_data
from cnn_model import CNNModel

# ✅ 하이퍼파라미터 설정
data_dir = "data_set"
seq_len = 20
batch_size = 32
epochs = 5
learning_rate = 0.001
model_path = "trained_model.pth"

# ✅ 데이터셋 로드
print("📂 `data_set/` 폴더 내 모든 JSON 파일을 불러오는 중...")
train_loader, dataset = load_data(data_dir, seq_len=seq_len, batch_size=batch_size)
vocab_size = dataset.vocab_size
print(f"📝 어휘 사전 크기: {vocab_size} 개 단어")

# ✅ 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(vocab_size=vocab_size).to(device)

# ✅ 기존 모델이 존재하면 불러와서 연장 학습 진행
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("🔄 기존 모델을 불러와 추가 학습을 진행합니다.")
else:
    print("🆕 새로운 모델을 생성하여 학습을 시작합니다.")

# ✅ 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ 학습 루프
model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"📢 Epoch {epoch}/{epochs} - 평균 손실: {avg_loss:.4f}")

# ✅ 학습 완료 후 모델 저장
torch.save(model.state_dict(), model_path)
print(f"✅ 학습된 모델이 '{model_path}' 파일로 저장되었습니다.")
