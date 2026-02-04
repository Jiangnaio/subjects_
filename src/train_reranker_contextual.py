# train_reranker_contextual.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 设置随机种子以确保可复现性
random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# ================= 配置 =================
CONFIG = {
    "sample_path": "./rerank_data_lora/rerank_train_samples_batched.json",
    "label_emb_path": "./rerank_data_lora/label_embeddings.pt",
    "query_emb_path": "./rerank_data_lora/query_embeddings.pt",
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "./rerank_data_lora/rerank_model_contextual.pth",
    "emb_dim": 768,
    "hidden_dim": 512,      # 可调：256 / 512
    "dropout": 0.1,
    "patience": 3,          # 早停耐心值
}

# ================= 数据集 =================
class BatchedRerankDataset(Dataset):
    def __init__(self, samples, query_embeddings, label_embeddings):
        self.samples = samples
        self.query_embs = query_embeddings.to(torch.float32)
        self.label_embs = label_embeddings.to(torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        q_idx = sample['query_idx']
        cand_indices = sample['candidate_label_indices']  # list of 500
        labels = torch.tensor(sample['labels'], dtype=torch.float32)  # (500,)
        query_emb = self.query_embs[q_idx]                # (D,)
        candidate_embs = self.label_embs[cand_indices]    # (500, D)
        return {
            'query_emb': query_emb,
            'candidate_embs': candidate_embs,
            'labels': labels
        }

# ================= 模型：Contextual Reranker =================
class ContextualReranker(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        input_dim = emb_dim * 3  # [c, c - mean, mean]

        # 投影到隐藏层（用于残差风格）
        self.proj_input = nn.Linear(input_dim, hidden_dim)
        # 双层MLP
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

        # 初始化：使用 He 初始化缓解梯度问题
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, query_emb, candidate_embs):
        """
        Args:
            query_emb: (B, D)
            candidate_embs: (B, K, D), K=500
        Returns:
            scores: (B, K)
        """
        B, K, D = candidate_embs.shape

        # 计算当前 batch 候选的均值（局部上下文）
        mean_emb = candidate_embs.mean(dim=1, keepdim=True)  # (B, 1, D)

        # 构造增强特征
        delta = candidate_embs - mean_emb                    # (B, K, D)
        expanded_mean = mean_emb.expand(-1, K, -1)           # (B, K, D)

        # 拼接: [原始, 偏移, 均值]
        enhanced = torch.cat([candidate_embs, delta, expanded_mean], dim=-1)  # (B, K, 3D)

        # 打分（仅基于标签上下文，轻量且有效）
        x = self.proj_input(enhanced)                        # (B, K, H)
        scores = self.mlp(x).squeeze(-1)                     # (B, K)

        return scores

# ================= 训练函数 =================
def train():
    print("Loading data...")
    with open(CONFIG["sample_path"], 'r', encoding='utf-8') as f:
        samples = json.load(f)
    label_embeddings = torch.load(CONFIG["label_emb_path"])
    query_embeddings = torch.load(CONFIG["query_emb_path"])

    dataset = BatchedRerankDataset(samples, query_embeddings, label_embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = ContextualReranker(
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        dropout=CONFIG["dropout"]
    ).to(CONFIG["device"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience_counter = 0

    print("Start training...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            q = batch['query_emb'].to(CONFIG["device"], non_blocking=True)
            c = batch['candidate_embs'].to(CONFIG["device"], non_blocking=True)
            y = batch['labels'].to(CONFIG["device"], non_blocking=True)

            optimizer.zero_grad()
            logits = model(q, c)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()
            total_loss += loss.item()
            

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")
        
        if epoch>=20 and epoch % 5==0:
            torch.save(model.state_dict(), CONFIG["model_save_path"].split('.pth')[0]+f'-epoch{epoch}'+'.pth')
            
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    print(f"✅ Training finished. Best model saved to {CONFIG['model_save_path']}")

if __name__ == "__main__":
    train()
