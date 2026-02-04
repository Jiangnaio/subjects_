# 3_train_from_embeddings.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_dim = 1024
expand_factor = 6

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, labels_csv):
        self.embeddings = torch.load(embeddings_path, weights_only=False)  # [N, 4096]
        self.df = pd.read_csv(labels_csv)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        labels = [int(x) for x in self.df.iloc[idx]["labels"].split()] if self.df.iloc[idx]["labels"] else []
        return {"embedding": emb, "labels": labels}

def collate_fn(batch):
    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {"embedding": embeddings, "labels": labels}

class LightXMLHead(nn.Module):
    """Only the trainable heads, no encoder"""
    def __init__(self, n_labels, n_clusters, hidden_dim=256, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        emb_dim = model_dim * expand_factor
        self.cluster_head = nn.Linear(emb_dim, n_clusters, dtype=dtype)
        self.bottleneck = nn.Linear(emb_dim, hidden_dim, dtype=dtype)
        self.label_embed = nn.Embedding(n_labels, hidden_dim)
        self.label_embed.weight.data = self.label_embed.weight.data.to(dtype)
        nn.init.xavier_uniform_(self.label_embed.weight)

    def forward(self, embeddings):
        embeddings = embeddings.to(self.dtype)
        cluster_logits = self.cluster_head(embeddings)
        proj = self.bottleneck(embeddings)
        return cluster_logits, proj
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--embed_dir", type=str, default="./embeddings")
    parser.add_argument("--model_dir", type=str, default="./model_from_emb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--expand_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--topk_clusters", type=int, default=10, help="Top-K clusters for candidate selection")
    parser.add_argument("--max_candidates", type=int, default=3000, help="Max number of candidate labels per sample")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    global expand_factor
    expand_factor = args.expand_factor
    
    device = torch.device("cuda")
    model_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    # Load data
    with open(os.path.join(args.data_dir, "full_label_map.json")) as f:
        n_labels = len(json.load(f))
    cluster_assign = np.load(os.path.join(args.data_dir, "cluster_assign.npy"))
    n_clusters = int(cluster_assign.max()) + 1

    # Build cluster_to_labels: list of label ids per cluster
    cluster_to_labels = [[] for _ in range(n_clusters)]
    for label_id, cid in enumerate(cluster_assign):
        if cid != -1:
            cluster_to_labels[cid].append(label_id)

    dataset = EmbeddingDataset(
        os.path.join(args.embed_dir, "train_embeddings.pt"),
        os.path.join(args.data_dir, "train.csv")
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    model = LightXMLHead(n_labels, n_clusters, hidden_dim=args.hidden_dim, dtype=model_dtype).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=(args.precision == "bf16"))
    criterion = nn.BCEWithLogitsLoss()
    # criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0, clip=0.05)

    train_losses = []
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            embeddings = batch["embedding"].to(device, non_blocking=True)
            labels_list = batch["labels"]

            cluster_logits, proj = model(embeddings)

            # --- Cluster loss (coarse-level) ---
            cluster_targets = torch.zeros(len(labels_list), n_clusters, device=device, dtype=cluster_logits.dtype)
            for i, lbs in enumerate(labels_list):
                for l in lbs:
                    cid = cluster_assign[l]
                    if cid != -1:
                        cluster_targets[i][cid] = 1.0
            loss_cluster = criterion(cluster_logits, cluster_targets)

            # --- Discriminator loss (fine-level) ---
            with torch.no_grad():
                cluster_probs = torch.sigmoid(cluster_logits)  # [B, n_clusters]
                topk_scores, topk_cids = torch.topk(cluster_probs, k=args.topk_clusters, dim=1)  # [B, K]

            loss_disc = 0.0
            valid_samples = 0
            for i in range(len(labels_list)):
                if len(labels_list[i]) == 0:
                    continue

                # Gather all labels from top-k clusters
                cand_set = set()
                for cid in topk_cids[i].cpu().tolist():
                    cand_set.update(cluster_to_labels[cid])
                if not cand_set:
                    continue

                # Convert to list and optionally subsample
                cand_list = list(cand_set)
                true_set = set(labels_list[i])

                # Ensure all positives are included
                pos_in_cand = [l for l in true_set] #正样本为所有真实标签
                neg_in_cand = [l for l in cand_list if l not in true_set]

                if len(cand_list) > args.max_candidates:
                    max_neg = args.max_candidates - len(pos_in_cand)
                    if max_neg > 0 and len(neg_in_cand) > max_neg:
                        neg_in_cand = np.random.choice(neg_in_cand, size=max_neg, replace=False).tolist()
                    cand_list = pos_in_cand + neg_in_cand
                else:
                    cand_list = pos_in_cand + neg_in_cand

                if not cand_list:
                    continue

                # Build target and compute scores
                cand_tensor = torch.tensor(cand_list, dtype=torch.long, device=device)
                label_embs = model.label_embed(cand_tensor)  # [K, H]
                scores = torch.matmul(label_embs, proj[i])   # [K]

                target = torch.tensor([1.0 if l in true_set else 0.0 for l in cand_list],
                                      device=device, dtype=scores.dtype)

                loss_disc += criterion(scores, target)
                valid_samples += 1

            if valid_samples > 0:
                loss_disc = loss_disc / valid_samples
            else:
                loss_disc = torch.tensor(0.0, device=device)

            loss = loss_cluster + loss_disc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"head-epoch{epoch}.pth"))

        # Plot loss
        plt.figure(figsize=(8,5))
        plt.plot(range(1, epoch+2), train_losses, marker='o')
        plt.title("Training Loss (From Embeddings)")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, f"loss_epoch{epoch}.png"))
        plt.close()

    torch.save(model.state_dict(), os.path.join(args.model_dir, "head.pth"))
    torch.save(cluster_assign, os.path.join(args.model_dir, "cluster_assign.pt"))
    print("✅ Training from embeddings done.")

if __name__ == "__main__":
    main()