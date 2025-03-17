import json
import os
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data_dir, seq_len):
        """
        `data_set/` 폴더 내 모든 JSON 파일을 불러와서 데이터셋을 생성.
        JSON 파일의 'tokens' 필드를 직접 사용하여 MeCab 없이 학습 가능.
        """
        self.seq_len = seq_len
        self.sentences = []

        # `data_set/` 폴더 내 모든 JSON 파일 불러오기
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

        if not json_files:
            raise FileNotFoundError(f"🚨 'data_set/' 폴더에 JSON 파일이 없습니다! {data_dir} 확인 필요")

        for filename in json_files:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sentences.extend([entry["tokens"] for entry in data["dataset"]])

        # ✅ 어휘 사전 생성
        unique_tokens = sorted(set(token for tokens in self.sentences for token in tokens))
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        vocab = [self.pad_token, self.unk_token] + unique_tokens
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)

        # ✅ 입력 시퀀스 및 타겟 생성
        self.X_data, self.y_data = [], []
        for tokens in self.sentences:
            if len(tokens) < seq_len + 1:
                continue  # 너무 짧은 문장은 제외

            for i in range(len(tokens) - seq_len):
                seq_tokens = tokens[i:i + seq_len]  # 입력 시퀀스
                target_token = tokens[i + seq_len]  # 다음 단어 예측
                seq_indices = [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in seq_tokens]
                target_index = self.word2idx.get(target_token, self.word2idx[self.unk_token])
                self.X_data.append(seq_indices)
                self.y_data.append(target_index)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return torch.tensor(self.X_data[idx], dtype=torch.long), torch.tensor(self.y_data[idx], dtype=torch.long)

def load_data(data_dir, seq_len=20, batch_size=32):
    """ `data_set/` 폴더 내 모든 JSON 파일에서 데이터를 불러와 DataLoader 반환 """
    dataset = TextDataset(data_dir, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset
