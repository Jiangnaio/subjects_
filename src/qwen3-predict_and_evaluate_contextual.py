 # predict_and_evaluate_contextual.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
device="cuda"
# ================= 模型定义（必须与训练一致） =================
class ContextualReranker(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        input_dim = emb_dim * 3
        self.proj_input = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query_emb, candidate_embs):
        """
        query_emb: (B, D)
        candidate_embs: (B, K, D)
        Returns: (B, K) logits
        """
        B, K, D = candidate_embs.shape
        mean_emb = candidate_embs.mean(dim=1, keepdim=True)  # (B, 1, D)
        delta = candidate_embs - mean_emb                    # (B, K, D)
        expanded_mean = mean_emb.expand(-1, K, -1)           # (B, K, D)
        enhanced = torch.cat([candidate_embs, delta, expanded_mean], dim=-1)  # (B, K, 3D)
        x = self.proj_input(enhanced)
        scores = self.mlp(x).squeeze(-1)
        return scores

# ================= 配置 =================
CONFIG = {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "lora_path": "/media/jh/新加卷1/2026/rag_code/qwen3-emb-4b-infonce-b16x3-lora48-pos8/checkpoints/checkpoint-267/",
    "max_length": 1500,
    "query_batch_size": 16,
    "rerank_query_batch_size": 128,      # 小 batch 避免 OOM（因 500×768）
    "retrieval_top_k": 500,
    "final_top_k": 500,
    "emb_dir": "./qwen3-rerank_data_lora",  # 注意：使用 batched 版本的数据
    "model_save_path": "./qwen3-rerank_data_lora/rerank_model_contextual-epoch170.pth",
    "dev_file": "/media/jh/新加卷1/2026/rag_code/data/test_data_all.json",
    "output_file": "./qwen3-rerank_data_lora/predictions_reranked_contextual.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "emb_dim": 2560, # qwen3-4b的hidden_dim为2560
}
TASK_DESCRIPTION = "Given a paper's title and abstract, retrieve relevant subject topics"

def initialize_model():
    from peft import get_peft_model, LoraConfig, TaskType
    dtype = torch.bfloat16
    # 初始化分词器 (左填充)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    # 加载基础模型
    print("Loading base model...")
    model = AutoModel.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    lora_path=CONFIG["lora_path"]
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.eval()
    model.to(device)
    return model, tokenizer
    
def last_token_pool(last_hidden_states, attention_mask):
    """获取最后一个非填充token的隐藏状态"""
    # 处理左填充情况
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]
        
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
        
        embeddings = last_token_pool(model(**inputs).last_hidden_state, inputs["attention_mask"])
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
        positive_gndids = [str(code).strip() for code in item.get('true_labels', [])]
        title = item['title']
        abstract = item['abstract']
        query = f'{title}: {abstract}'
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
        # Step 1: coarse retrieval
        sim = torch.mm(q_batch, label_embs.t())  # (B, L)
        _, topk_indices = torch.topk(sim, k=top_k_retrieve, dim=1, largest=True)  # (B, K)
        candidate_embs = label_embs[topk_indices]  # (B, K, D)
        # Step 2: rerank using contextual model
        scores = reranker(q_batch, candidate_embs)  # (B, K)
        # Step 3: select top final
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

# ================= 主函数 =================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_debug", action="store_true", 
                        help="test with a small subset of data")
    args = parser.parse_args()

    print("Initializing encoder...")
    encoder, tokenizer = initialize_model()

    print("Loading reranker model...")
    reranker = ContextualReranker(
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=512,
        dropout=0.1
    ).to(CONFIG["device"])
    reranker.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))
    reranker.eval()

    print("Loading embeddings and label codes...")
    label_embeddings = torch.load(os.path.join(CONFIG["emb_dir"], "label_embeddings.pt")).to(torch.float32)
    with open(os.path.join(CONFIG["emb_dir"], "label_codes.json"), 'r', encoding='utf-8') as f:
        label_codes = json.load(f)

    print("Loading dev data...")
    dev_data = load_dev_data(CONFIG["dev_file"])
    if args.use_debug:
        dev_data = dev_data[:2000]
    queries = [f"Instruct: {TASK_DESCRIPTION}\nQuery: {item['query']}" for item in dev_data]
    ground_truths = [set(item['positive_gndids']) for item in dev_data]

    query_emb_path='/media/jh/新加卷1/2026/rag_code/cache/query_embeddings-s267.pkl'
    if os.path.exists(query_emb_path):
        import pickle
        with open(query_emb_path, 'rb') as f:
            query_embeddings = pickle.load(f)    
        query_embeddings = torch.tensor(query_embeddings).to(torch.float32).to(CONFIG["device"])
        print(f'✅ Load query embeddings from {query_emb_path}')
    else:
        print("Encoding queries...")
        query_embeddings = encode_texts(queries, encoder, tokenizer, CONFIG["query_batch_size"], "")

    # Coarse retrieval (for comparison)
    print("Computing coarse retrieval results...")
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
    print("Computing reranked results...")
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
    print(f"✅ Reranked predictions saved to {CONFIG['output_file']}")
    

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
