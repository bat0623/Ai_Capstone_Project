import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=[3, 4, 5], dropout_rate=0.5):
        """CNN 기반 언어 모델 정의"""
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # 여러 가지 커널 크기를 사용하는 컨볼루션 레이어 리스트
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            ) for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), vocab_size)

    def forward(self, x):
        """입력: (batch_size, seq_len) 형태의 단어 인덱스"""
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)

        conv_results = []
        for conv in self.convs:
            c_out = conv(x)  # Conv -> BatchNorm -> ReLU
            c_out, _ = torch.max(c_out, dim=2)  # 글로벌 맥스 풀링
            conv_results.append(c_out)

        out = torch.cat(conv_results, dim=1)
        out = self.dropout(out)
        out = self.fc(out)  # Fully Connected Layer
        return out
