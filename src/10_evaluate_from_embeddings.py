import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse
import random
MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'
model_dim = 1024
expand_factor = 4
# ----------------------------
# Dataset for Precomputed Embeddings
# ----------------------------
class EmbeddingTestDataset(Dataset):
    """Dataset that loads precomputed embeddings instead of raw text"""
    def __init__(self, embeddings_path, csv_file):
        # Load embeddings: [N, 4096] tensor
        self.embeddings = torch.load(embeddings_path, weights_only=False)
        self.df = pd.read_csv(csv_file)
        assert len(self.embeddings) == len(self.df), f"Embedding count mismatch{len(self.embeddings)},{len(self.df)}"

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        row = self.df.iloc[idx]
        labels = [int(x) for x in row["labels"].split()] if row["labels"] else []
        return {
            "embedding": embedding,
            "labels": labels
        }

def collate_fn(batch):
    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {
        "embeddings": embeddings,
        "labels": labels
    }

# ----------------------------
# LightXML Head Only (No Encoder)
# ----------------------------
class LightXMLHead(nn.Module):
    """Only the trainable heads, no encoder - matches training setup"""
    def __init__(self, n_labels, n_clusters, hidden_dim=256, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        emb_dim = model_dim * expand_factor
        
        # Cluster head: predicts cluster probabilities
        self.cluster_head = nn.Linear(emb_dim, n_clusters, dtype=dtype)
        
        # Bottleneck: projects to label embedding space
        self.bottleneck = nn.Linear(emb_dim, hidden_dim, dtype=dtype)
        
        # Label embeddings: learnable label representations
        self.label_embed = nn.Embedding(n_labels, hidden_dim) #一个查找表，包含所有标签的嵌入向量
        self.label_embed.weight.data = self.label_embed.weight.data.to(dtype)
        nn.init.xavier_uniform_(self.label_embed.weight)

    def forward(self, embeddings):
        """Forward pass using precomputed embeddings"""
        embeddings = embeddings.to(self.dtype)
        cluster_logits = self.cluster_head(embeddings)  # [B, n_clusters]
        proj = self.bottleneck(embeddings)              # [B, hidden_dim]
        return cluster_logits, proj

# ----------------------------
# Evaluation Metrics
# ----------------------------
def compute_metrics_at_k(preds, trues, k):
    if not trues:
        return None, None, None
    topk_preds = preds[:k]
    true_set = set(trues)
    pred_set = set(topk_preds)
    hits = len(true_set & pred_set)
    precision = hits / min(k, len(pred_set))
    recall = hits / len(true_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# ----------------------------
# Main Evaluation Loop
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data-test")
    parser.add_argument("--model_dir", type=str, default="./model_from_emb")
    parser.add_argument("--embed_dir", type=str, default="./embeddings",
                        help="Directory containing precomputed embeddings")
    parser.add_argument("--batch_size", type=int, default=512,  # Much larger possible!
                        help="Batch size for inference (can be large since no encoder)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--use_test", action="store_true", 
                        help="Use test.csv instead of train subset")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Inference precision")
    parser.add_argument("--subset_size", type=int, default=1000,
                        help="Size of random subset when not using full test set")
    parser.add_argument("--expand_factor", type=int, default=4)
    parser.add_argument("--model_dim", type=int, default=1024)
    args = parser.parse_args()
    global expand_factor, model_dim
    expand_factor, model_dim = args.expand_factor, args.model_dim

    # Check BF16 support
    if args.precision == "bf16":
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 not supported"
        model_dtype = torch.bfloat16
        print("✅ Using bfloat16 inference (embedding-based)")
    else:
        model_dtype = torch.float32
        print("✅ Using float32 inference (embedding-based)")

    device = torch.device("cuda")

    # Load label map and cluster assignment
    with open(os.path.join(args.data_dir, "full_label_map.json")) as f:
        gnd_to_idx = json.load(f)
    n_labels = len(gnd_to_idx)
    cluster_assign = torch.load(os.path.join(args.model_dir, "cluster_assign.pt"), weights_only=False)
    n_clusters = int(cluster_assign.max()) + 1

    # Determine which embeddings to use
    if args.use_test:
        csv_file = os.path.join(args.data_dir, "test.csv")
        embeddings_path = os.path.join(args.embed_dir, "test_embeddings.pt")
        print(f"Using test set: {csv_file}")
    else:
        csv_file = os.path.join(args.data_dir, "train.csv")
        embeddings_path = os.path.join(args.embed_dir, "train_embeddings.pt")
        print(f"Using train subset (size={args.subset_size})")

    # Create dataset
    full_dataset = EmbeddingTestDataset(embeddings_path, csv_file)
    
    # Use subset if not using full test set
    if not args.use_test:
        indices = random.sample(range(len(full_dataset)), min(args.subset_size, len(full_dataset)))
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Load ONLY the head model (no encoder needed)
    model = LightXMLHead(n_labels, n_clusters, hidden_dim=args.hidden_dim, dtype=model_dtype).to(device)
    
    # Load weights - note: we now load the head weights, not full model
    head_path = os.path.join(args.model_dir, "head.pth")  # Default to final model
    if not os.path.exists(head_path):
        # 如果没有 final model，再 fallback 到 last epoch
        head_path = sorted([f for f in os.listdir(args.model_dir) if f.startswith("head-epoch")])[-1]
        head_path = os.path.join(args.model_dir, head_path)
    
    print(f"Loading head weights from: {head_path}")
    state_dict = torch.load(head_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.eval()

    all_preds = []   # Predicted label indices (sorted by score)
    all_trues = []   # Ground truth label indices

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            embeddings = batch["embeddings"].to(device, non_blocking=True)
            true_labels = batch["labels"]

            # Forward pass through head only
            with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(args.precision == "bf16")):
                cluster_logits, proj = model(embeddings)
                cluster_probs = torch.sigmoid(cluster_logits)  # [B, n_clusters]

            # Process each sample in batch
            for i in range(len(true_labels)):
                # Get top clusters (e.g., top 10)
                top_clusters = torch.topk(cluster_probs[i], k=10).indices.cpu().numpy()
                
                # Collect candidate labels from top clusters
                candidates = set()
                for cid in top_clusters:#收集指定簇的所有标签索引。
                    mask = (cluster_assign == cid)
                    candidates.update(np.where(mask)[0].tolist())
                candidates = list(candidates)
                
                if not candidates:
                    all_preds.append([])
                    all_trues.append(true_labels[i])
                    continue

                # Score candidate labels
                cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(args.precision == "bf16")):
                    cand_emb = model.label_embed(cand_tensor)  # 索引查找操作，从label_embed中查找对应的嵌入向量，[K, hidden_dim]
                    scores = torch.matmul(cand_emb, proj[i])   # [K]
                
                # Get top predictions
                sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
                pred_labels = [candidates[idx] for idx in sorted_indices]
                all_preds.append(pred_labels)
                all_trues.append(true_labels[i])

    # Evaluate for multiple k values
    k_list = [5, 10, 20, 30, 50,100,500]
    results = {k: {"precisions": [], "recalls": [], "f1s": []} for k in k_list}

    for preds, trues in zip(all_preds, all_trues):
        for k in k_list:
            p, r, f1 = compute_metrics_at_k(preds, trues, k)
            if p is not None:  # Skip samples with no true labels
                results[k]["precisions"].append(p)
                results[k]["recalls"].append(r)
                results[k]["f1s"].append(f1)

    # Print results in formatted table
    print("\n" + "="*80)
    print(f"{'Evaluation Results (Embedding-Based Inference)':^80}")
    print("="*80)
    print(f"{'k':<5} {'Precision@k':<15} {'Recall@k':<15} {'F1@k':<15} {'Samples':<10}")
    print("-"*80)
    for k in k_list:
        prec_list = results[k]["precisions"]
        rec_list = results[k]["recalls"]
        f1_list = results[k]["f1s"]
        
        prec = np.mean(prec_list) if prec_list else 0.0
        rec = np.mean(rec_list) if rec_list else 0.0
        f1 = np.mean(f1_list) if f1_list else 0.0
        n_samples = len(prec_list)
        
        print(f"{k:<5} {prec:<15.4f} {rec:<15.4f} {f1:<15.4f} {n_samples:<10}")
    print("="*80)
    print(f"✅ Evaluation done on {len(all_trues)} samples.")

if __name__ == "__main__":
    main()