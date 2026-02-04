import os
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse
import random

class TestDataset(Dataset):
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

class LightXMLModel(nn.Module):
    def __init__(self, n_labels, n_clusters, hidden_dim=256):
        super().__init__()
        # self.roberta = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
        model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
        
        # Load base model in specified dtype
        self.roberta = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16  # ← key change
        )
        for p in self.roberta.parameters():
            p.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        emb_dim = 4 * 1024
        self.cluster_head = nn.Linear(emb_dim, n_clusters)
        self.bottleneck = nn.Linear(emb_dim, hidden_dim)
        self.label_embed = nn.Embedding(n_labels, hidden_dim)
        nn.init.xavier_uniform_(self.label_embed.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_embs = torch.cat([outputs.hidden_states[-i][:, 0] for i in range(1, 5)], dim=-1)
        cls_embs = self.dropout(cls_embs)
        cluster_logits = self.cluster_head(cls_embs)
        proj = self.bottleneck(cls_embs)
        return cluster_logits, proj

def compute_metrics_at_k(preds, trues, k):
    """
    Compute Precision@k, Recall@k, F1@k for one sample.
    preds: list of predicted label indices (sorted by score, descending)
    trues: list of ground truth label indices
    """
    if not trues:
        # Skip samples with no true labels (common in XML)
        return None, None, None

    topk_preds = preds[:k]
    true_set = set(trues)
    pred_set = set(topk_preds)

    hits = len(true_set & pred_set)
    precision = hits / min(k, len(pred_set)) if pred_set else 0.0  # but pred_set always size k>0
    recall = hits / len(true_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_test", action="store_true", help="Use test.csv instead of train subset")
    args = parser.parse_args()

    # Load label map and cluster assignment
    with open(os.path.join(args.data_dir, "full_label_map.json")) as f:
        gnd_to_idx = json.load(f)
    n_labels = len(gnd_to_idx)
    cluster_assign = torch.load(os.path.join(args.model_dir, "cluster_assign.pt"), weights_only=False)
    n_clusters = int(cluster_assign.max()) + 1

    # Load tokenizer and dataset
    # tokenizer = XLMRobertaTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if args.use_test:
        csv_file = os.path.join(args.data_dir, "test.csv")
        dataset = TestDataset(csv_file, tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        # Use train subset for quick evaluation (as in original)
        csv_file = os.path.join(args.data_dir, "train.csv")
        dataset = TestDataset(csv_file, tokenizer)
        indices = random.sample(range(len(dataset)), min(1000, len(dataset)))
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = LightXMLModel(n_labels, n_clusters)
    state_dict = torch.load(os.path.join(args.model_dir, "model.pth"))
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    all_preds = []   # each element: list of predicted label indices (sorted by score)
    all_trues = []   # each element: list of true label indices

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            true_labels = batch["labels"]

            cluster_logits, proj = model(input_ids, attention_mask)
            cluster_probs = torch.sigmoid(cluster_logits)  # [B, C]

            for i in range(len(true_labels)):
                # Get top clusters (e.g., top 10 clusters)
                top_clusters = torch.topk(cluster_probs[i], k=10).indices.cpu().numpy()
                candidates = set()
                for cid in top_clusters:
                    mask = (cluster_assign == cid)
                    candidates.update(np.where(mask)[0].tolist())
                candidates = list(candidates)
                if not candidates:
                    all_preds.append([])
                    all_trues.append(true_labels[i])
                    continue

                # Score all candidate labels
                cand_tensor = torch.tensor(candidates, dtype=torch.long).cuda()
                cand_emb = model.label_embed(cand_tensor)  # [K, H]
                scores = torch.matmul(cand_emb, proj[i])   # [K]
                # Sort candidates by score (descending)
                sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
                pred_labels = [candidates[idx] for idx in sorted_indices]
                all_preds.append(pred_labels)
                all_trues.append(true_labels[i])

    # Evaluate for multiple k values
    k_list = [5, 10, 20, 30, 50]
    results = {k: {"precisions": [], "recalls": [], "f1s": []} for k in k_list}

    for preds, trues in zip(all_preds, all_trues):
        for k in k_list:
            p, r, f1 = compute_metrics_at_k(preds, trues, k)
            if p is not None:  # skip if no true labels
                results[k]["precisions"].append(p)
                results[k]["recalls"].append(r)
                results[k]["f1s"].append(f1)

    # Print results
    print("\n" + "="*60)
    print(f"{'k':<5} {'Precision@k':<15} {'Recall@k':<15} {'F1@k':<15}")
    print("="*60)
    for k in k_list:
        prec = np.mean(results[k]["precisions"]) if results[k]["precisions"] else 0.0
        rec = np.mean(results[k]["recalls"]) if results[k]["recalls"] else 0.0
        f1 = np.mean(results[k]["f1s"]) if results[k]["f1s"] else 0.0
        print(f"{k:<5} {prec:<15.4f} {rec:<15.4f} {f1:<15.4f}")
    print("="*60)

    print("✅ Evaluation done.")

if __name__ == "__main__":
    main()