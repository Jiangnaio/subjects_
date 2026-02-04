import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt

# ----------------------------
# Dataset & Collate (unchanged)
# ----------------------------
class XMCDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        labels = [int(x) for x in row["labels"].split()] if row["labels"] else []
        inputs = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ----------------------------
# Model Definition
# ----------------------------
class LightXMLModel(nn.Module):
    def __init__(self, n_labels, n_clusters, hidden_dim=256, model_dtype=torch.float32):
        super().__init__()
        model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
        
        # Load base model in specified dtype
        self.roberta = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=model_dtype  # ← key change
        )
        for p in self.roberta.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        emb_dim = 4 * 1024  # arctic-embed-l uses 1024 hidden size
        self.cluster_head = nn.Linear(emb_dim, n_clusters).to(model_dtype)
        self.bottleneck = nn.Linear(emb_dim, hidden_dim).to(model_dtype)
        self.label_embed = nn.Embedding(n_labels, hidden_dim).to(model_dtype)
        nn.init.xavier_uniform_(self.label_embed.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Concatenate last 4 layers' [CLS] token
        cls_embs = torch.cat([outputs.hidden_states[-i][:, 0] for i in range(1, 5)], dim=-1)
        cls_embs = self.dropout(cls_embs)
        cluster_logits = self.cluster_head(cls_embs)
        proj = self.bottleneck(cls_embs)
        return cluster_logits, proj

# ----------------------------
# Negative Sampling (unchanged)
# ----------------------------
def dynamic_negative_sampling(cluster_assign, true_labels, n_candidates=2000):
    true_labels = set(true_labels)
    cluster_ids = set(cluster_assign[l] for l in true_labels if cluster_assign[l] != -1)
    candidates = set()
    for cid in cluster_ids:
        mask = (cluster_assign == cid)
        candidates.update(np.where(mask)[0].tolist())
    while len(candidates) < n_candidates:
        neg = np.random.randint(0, len(cluster_assign))
        if neg not in true_labels:
            candidates.add(neg)
    return list(candidates)

# ----------------------------
# Main Training Loop
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Training precision: fp32 or bf16")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # Set device and dtype
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    
    if args.precision == "bf16":
        model_dtype = torch.bfloat16
        assert torch.cuda.is_bf16_supported(), "BF16 not supported on this GPU"
        print("✅ Using bfloat16 precision (full model in bf16)")
    else:
        model_dtype = torch.float32
        print("✅ Using float32 precision")

    # Load data
    with open(os.path.join(args.data_dir, "full_label_map.json")) as f:
        n_labels = len(json.load(f))
    cluster_assign = np.load(os.path.join(args.data_dir, "cluster_assign.npy"))
    n_clusters = int(cluster_assign.max()) + 1

    model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = XMCDataset(os.path.join(args.data_dir, "train.csv"), tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model in target dtype
    model = LightXMLModel(n_labels, n_clusters, hidden_dim=256, model_dtype=model_dtype).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.cluster_head.parameters(), "lr": args.lr},
        {"params": model.bottleneck.parameters(), "lr": args.lr},
        {"params": model.label_embed.parameters(), "lr": args.lr}
    ], fused=True if args.precision == "bf16" else False)  # fused AdamW supports bf16

    criterion = nn.BCEWithLogitsLoss()

    # For logging
    train_losses = []
    learning_rates = []

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)          # long tensor
            attention_mask = batch["attention_mask"].to(device)  # long tensor
            labels = batch["labels"]

            # Forward pass in native precision (no autocast)
            cluster_logits, proj = model(input_ids, attention_mask)

            # Build cluster targets in same dtype as logits
            cluster_targets = torch.zeros(len(labels), n_clusters, device=device, dtype=cluster_logits.dtype)
            for i, lbs in enumerate(labels):
                for l in lbs:
                    cid = cluster_assign[l]
                    if cid != -1:
                        cluster_targets[i][cid] = 1.0

            loss_cluster = criterion(cluster_logits, cluster_targets)

            # Discriminator loss
            loss_disc = 0.0
            for i in range(len(labels)):
                if len(labels[i]) == 0:
                    continue
                cand_labels = dynamic_negative_sampling(cluster_assign, labels[i], n_candidates=500)
                cand_tensor = torch.tensor(cand_labels, dtype=torch.long, device=device)
                cand_emb = model.label_embed(cand_tensor)  # [K, H]
                scores = torch.matmul(cand_emb, proj[i])   # [K]
                target = torch.zeros(len(cand_labels), device=device, dtype=scores.dtype)
                for j, l in enumerate(cand_labels):
                    if l in labels[i]:
                        target[j] = 1.0
                loss_disc += criterion(scores, target)

            loss = loss_cluster + loss_disc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Save checkpoint in native precision
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"model-epoch{epoch}.pth"))

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        # Visualization (unchanged)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epoch + 2), train_losses, marker='o', label='Training Loss')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, f'loss_curve_epoch{epoch}.png'))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epoch + 2), learning_rates, marker='s', color='orange', label='Learning Rate')
        plt.title('Learning Rate per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, f'lr_curve_epoch{epoch}.png'))
        plt.close()

    # Final save
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    torch.save(cluster_assign, os.path.join(args.model_dir, "cluster_assign.pt"))
    print("✅ Training done.")

if __name__ == "__main__":
    main()