# predict_and_evaluate_batched.py
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ------------------ 模型定义（需与训练一致） ------------------
class AllAtOnceReranker(torch.nn.Module):
    def __init__(self, emb_dim=768, num_candidates=500, hidden_dim=256, num_heads=8):
        super().__init__()
        self.query_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.cand_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, query_emb, candidate_embs):
        B, K, D = candidate_embs.shape
        q = self.query_proj(query_emb).unsqueeze(1)
        c = self.cand_proj(candidate_embs)
        attn_out, _ = self.cross_attn(query=q, key=c, value=c)
        attn_query = attn_out.squeeze(1)
        expanded_query = attn_query.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([expanded_query, c], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores

# ------------------ 配置 ------------------
CONFIG = {
    "model_name": "/media/jh/新加卷/11_15/codes/arctic-emb-m-损失函数向量化-b48/final/",
    "max_length": 1500,
    "query_batch_size": 16,
    "rerank_query_batch_size": 128,  # smaller due to memory
    "retrieval_top_k": 500,
    "final_top_k": 50,
    "emb_dir": "./rerank_data_lora",
    "dev_file": "/media/jh/新加卷/11_15/codes/datasets/qwen3_embedding_dev.json",
    "output_file": "./rerank_data_lora/predictions_reranked_lora.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "emb_dim": 768,
}
TASK_DESCRIPTION = "Given a paper's title and abstract, retrieve relevant subject topics"

def initialize_encoder():
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if hasattr(config, "use_memory_efficient_attention"):
        config.use_memory_efficient_attention = False
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModel.from_pretrained(
        CONFIG["model_name"],
        config=config,
        dtype=torch.bfloat16,
        add_pooling_layer=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = model.to(CONFIG["device"])
    model.eval()
    return model, tokenizer

@torch.no_grad()
def encode_texts(texts, model, tokenizer, batch_size=8, prefix=""):
    all_embeddings = []
    texts = [f"{prefix}{t}" for t in texts]
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        ).to(CONFIG["device"])
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1).to(torch.float32)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

def load_dev_data(dev_file):
    with open(dev_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    processed_data = []
    for item in data:
        positive_gndids = [str(code).strip() for code in item.get('positive_gndids', [])]
        query = item.get('query', '')
        title = query.split('Abstract:')[0].split('Title:')[1].strip() if 'Title:' in query else ""
        abstract = query.split('Abstract:')[1].strip() if 'Abstract:' in query else ""
        processed_item = {
            'query': query,
            'title': title,
            'abstract': abstract,
            'positive_gndids': positive_gndids
        }
        processed_data.append(processed_item)
    return processed_data

@torch.no_grad()
def rerank_all_at_once(query_embs, label_embs, reranker, label_codes, top_k_retrieve=500, top_k_final=50, batch_size=8):
    device = query_embs.device
    label_embs = label_embs.to(device)
    reranker.eval()
    all_predictions = []
    N = query_embs.size(0)
    for i in tqdm(range(0, N, batch_size), desc="Reranking"):
        q_batch = query_embs[i:i+batch_size].to(device)  # (B, D)
        sim = torch.mm(q_batch, label_embs.t())  # (B, L)
        _, topk_indices = torch.topk(sim, k=top_k_retrieve, dim=1, largest=True)  # (B, K)
        candidate_embs = label_embs[topk_indices]  # (B, K, D)
        scores = reranker(q_batch, candidate_embs)  # (B, K)
        _, rerank_idx = torch.topk(scores, k=top_k_final, dim=1, largest=True)
        final_indices = topk_indices.gather(1, rerank_idx)  # (B, final_k)
        for j in range(final_indices.size(0)):
            codes = [label_codes[idx.item()] for idx in final_indices[j]]
            all_predictions.append(codes)
    return all_predictions

def compute_metrics(predictions, ground_truths, ks=[5,10,20,30,50]):
    metrics = {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for k in ks}
    n_valid = 0
    for pred_list, gt_set in zip(predictions, ground_truths):
        if not gt_set:
            continue
        n_valid += 1
        for k in ks:
            topk_pred = pred_list[:k]
            hits = len(set(topk_pred) & gt_set)
            precision = hits / k
            recall = hits / len(gt_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics[k]["precision"] += precision
            metrics[k]["recall"] += recall
            metrics[k]["f1"] += f1
    if n_valid == 0:
        return metrics, 0
    for k in ks:
        metrics[k]["precision"] /= n_valid
        metrics[k]["recall"] /= n_valid
        metrics[k]["f1"] /= n_valid
    return metrics, n_valid

def main():
    # Load data
    encoder, tokenizer = initialize_encoder()
    reranker = AllAtOnceReranker(emb_dim=CONFIG["emb_dim"]).to(CONFIG["device"])
    reranker.load_state_dict(torch.load("./rerank_model_batched.pth", map_location=CONFIG["device"]))
    reranker.eval()

    label_embeddings = torch.load(os.path.join(CONFIG["emb_dir"], "label_embeddings.pt")).to(torch.float32)
    with open(os.path.join(CONFIG["emb_dir"], "label_codes.json"), 'r', encoding='utf-8') as f:
        label_codes = json.load(f)

    dev_data = load_dev_data(CONFIG["dev_file"])
    queries = [f"Instruct: {TASK_DESCRIPTION}\nQuery: {item['query']}" for item in dev_data]
    ground_truths = [set(item['positive_gndids']) for item in dev_data]

    # Encode queries
    query_embeddings = encode_texts(queries, encoder, tokenizer, CONFIG["query_batch_size"], "")

    # Coarse retrieval (for comparison)
    device = query_embeddings.device
    label_embs_gpu = label_embeddings.to(device)
    coarse_predictions = []
    for i in tqdm(range(0, len(query_embeddings), CONFIG["rerank_query_batch_size"]), desc="Coarse retrieval"):
        q_batch = query_embeddings[i:i+CONFIG["rerank_query_batch_size"]].to(device)
        sim = torch.mm(q_batch, label_embs_gpu.t())
        _, topk_idx = torch.topk(sim, k=CONFIG["final_top_k"], dim=1, largest=True)
        for idxs in topk_idx:
            codes = [label_codes[idx.item()] for idx in idxs]
            coarse_predictions.append(codes)

    # Reranked predictions
    reranked_predictions = rerank_all_at_once(
        query_embs=query_embeddings,
        label_embs=label_embeddings,
        reranker=reranker,
        label_codes=label_codes,
        top_k_retrieve=CONFIG["retrieval_top_k"],
        top_k_final=CONFIG["final_top_k"],
        batch_size=CONFIG["rerank_query_batch_size"]
    )

    # Save
    with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
        json.dump(reranked_predictions, f, indent=2)

    # Evaluate
    ks = [5, 10, 20, 30, 50]
    coarse_metrics, n_valid = compute_metrics(coarse_predictions, ground_truths, ks)
    rerank_metrics, _ = compute_metrics(reranked_predictions, ground_truths, ks)

    print("\n" + "="*80)
    print("Evaluation Results (P@k, R@k, F1@k)")
    print("="*80)
    print(f"{'k':<5} {'Method':<12} {'Precision@k':<14} {'Recall@k':<12} {'F1@k':<12}")
    print("-"*80)
    for k in ks:
        c_p = coarse_metrics[k]['precision']
        c_r = coarse_metrics[k]['recall']
        c_f1 = coarse_metrics[k]['f1']
        r_p = rerank_metrics[k]['precision']
        r_r = rerank_metrics[k]['recall']
        r_f1 = rerank_metrics[k]['f1']
        print(f"{k:<5} {'Coarse':<12} {c_p:<14.4f} {c_r:<12.4f} {c_f1:<12.4f}")
        print(f"{k:<5} {'Reranked':<12} {r_p:<14.4f} {r_r:<12.4f} {r_f1:<12.4f}")
        dp = r_p - c_p
        dr = r_r - c_r
        df1 = r_f1 - c_f1
        print(f"{'':<5} {'Δ':<12} {dp:<+14.4f} {dr:<+12.4f} {df1:<+12.4f}")
    print("-"*80)
    print(f"\nValid samples: {n_valid}/{len(dev_data)}")
    print("✅ Evaluation completed.")

if __name__ == "__main__":
    main()
