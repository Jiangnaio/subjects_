# -*- coding: utf-8 -*-
"""
融合向量检索与 LightXML 的结果（RRF 融合）
- 向量检索：top-500
- LightXML：top-200
- 融合方式：Reciprocal Rank Fusion (RRF)
- 评估指标：Recall@k, Precision@k, F1@k for k in [5,10,20,30,40,50]
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import argparse

# ================= 配置 =================
CONFIG = {
    "model_name": "/media/jh/新加卷/11_15/codes/arctic-embed-l-full-ft/best_model/",
    "lora_path": None,
    "max_length": 1500,
    "encoding_batch_size": 32,
    "query_batch_size": 8,
    "vector_top_k": 500,     # 向量检索数量
    "lightxml_top_k": 500,   # LightXML 检索数量
    "final_top_k": 500,       # 最终输出数量
    "emb_dir": "./arctic-l-ft-emb1",
    "dev_file": "/media/jh/新加卷1/2026/rag_code/data/test_data_all.json",
    "label_file": "/media/jh/新加卷/11_15/llms4subjects-1.0.1/shared-task-datasets/GND/dataset/GND-Subjects-all.json",
    "prefix": "arctic-embed-l-v2-full_finetune-withot-hard",
    "debug_mode": False,
    "debug_samples": 3,
}

# ================= 工具函数 =================
def load_label_data(gnd_to_idx):
    with open(CONFIG["label_file"], 'r', encoding='utf-8') as f:
        gnd_jsons = json.load(f)
    print('gnd_jsons',gnd_jsons[5])
    labels = []
    for c in gnd_jsons:
        classification = c.get("Classification Name", "Unknown")
        class_num = str(c['Classification Number'])
        name = c.get("Name", "Unknown")
        alt_names = c.get("Alternate Name", [])
        if alt_names != []:
            alt_names = ', '.join(alt_names)
        rs=c.get('Related Subjects',[])
        definition = c.get("Definition", "")
        if rs != []:
            rs = ', '.join(rs)
        gnd_code = str(c['Code']).split(':')[-1]
        if gnd_code not in gnd_to_idx:
            raise ValueError(f"Invalid GND code: {gnd_code}, not in gnd_to_idx")
        
        passage_text = f"Classification: {classification}. Subject: {name}. Alternate Names: {alt_names}. Related: {rs}. Definition: {definition}"
        
        labels.append({
            'sentence2': passage_text,
            'Code': gnd_code,
            'name': name,
            'alt_names': alt_names,
        })
    print('labbels',labels[0])
    return labels

def load_dev_data(dev_file):
    data=pd.read_csv(dev_file)
    processed_data = []
    for text,labels in zip(data['text'],data["labels"]):
        labels = [int(x) for x in labels.split()]
        query=f"""Instruct: Given a paper's title and abstract, retrieve relevant subject topics\nQuery:{text}"""
        processed_data.append({
            'query': query,
            "labels": labels,
        })
    return processed_data

def initialize_model():
    """初始化 Arctic Embed 模型"""
    print(f"Loading model: {CONFIG['model_name']}")
        # Arctic-Emb-m-v2需要特殊处理
    from transformers import AutoConfig  # 新增
    # 1. 先加载 config
    config = AutoConfig.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True
    )
    
    # 2. 强制关闭 memory_efficient_attention（关键！）
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
    if CONFIG["lora_path"] and os.path.exists(CONFIG["lora_path"]):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, CONFIG["lora_path"])
        # 合并lora
        model=model.merge_and_unload()
        torch.cuda.empty_cache()
        print(f"Loaded LoRA weights from {CONFIG['lora_path']}")
    model = model.cuda()
    model.eval()
    return model, tokenizer

def encode_texts(texts, model, tokenizer, batch_size, prefix=""):
    device = next(model.parameters()).device
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {prefix}"):
        batch = [f"{prefix}{t}" for t in texts[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=CONFIG["max_length"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index

def reciprocal_rank_fusion(list1, list2, k1=2, k2=2):
    """
    Reciprocal Rank Fusion (RRF): score = 1 / (k + rank)
    """
    scores = defaultdict(float)
    for rank, code in enumerate(list1, 1):
        scores[code] += 1.0 / (k1 + rank)
    for rank, code in enumerate(list2, 1):
        scores[code] += 1.0 / (k2 + rank)
    sorted_codes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_codes

def evaluate_metrics(pred_list, gt_list, ks=[5,10,20,30,40,50]):
    metrics = {k: {"recall": 0.0, "precision": 0.0, "f1": 0.0} for k in ks}
    n_valid = 0
    for pred, gt in zip(pred_list, gt_list):
        gt_set = set(gt)
        if not gt_set:
            continue
        n_valid += 1
        for k in ks:
            topk = pred[:k]
            hits = len(set(topk) & gt_set)
            recall = hits / len(gt_set)
            precision = hits / k
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
            metrics[k]["recall"] += recall
            metrics[k]["precision"] += precision
            metrics[k]["f1"] += f1
    if n_valid == 0:
        return metrics, 0
    for k in ks:
        metrics[k]["recall"] /= n_valid
        metrics[k]["precision"] /= n_valid
        metrics[k]["f1"] /= n_valid
    return metrics, n_valid

# ================= LightXML 相关 =================
class EmbeddingTestDataset(Dataset):
    def __init__(self, embeddings_path, csv_file):
        self.embeddings = torch.load(embeddings_path, weights_only=False)
        self.df = pd.read_csv(csv_file)
        assert len(self.embeddings) == len(self.df)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        row = self.df.iloc[idx]
        labels = [int(x) for x in row["labels"].split()] if row["labels"] else []
        return {"embedding": embedding, "labels": labels}

def collate_fn(batch):
    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {"embeddings": embeddings, "labels": labels}

class LightXMLHead(nn.Module):
    def __init__(self, n_labels, n_clusters, hidden_dim=256, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        emb_dim = 1024 * 6  # expand_factor = 4
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

def run_lightxml_inference(args, dev_data, label_metadata, gnd_to_idx, idx_to_gnd):
    # 构建 fake CSV 和 embedding 文件（仅用于兼容）
    # 实际中应提前生成 test.csv 和 test_embeddings.pt
    dataset = EmbeddingTestDataset(args.embeddings_path, args.csv_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    with open(os.path.join(args.data_dir, "full_label_map.json")) as f:
        gnd_to_idx = json.load(f)
    idx_to_gnd = {v: k for k, v in gnd_to_idx.items()}
    n_labels = len(gnd_to_idx)
    cluster_assign = torch.load(os.path.join(args.model_dir, "cluster_assign.pt"), weights_only=False)
    n_clusters = int(cluster_assign.max()) + 1

    model = LightXMLHead(n_labels, n_clusters, hidden_dim=args.hidden_dim, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "head.pth"), map_location="cpu"))
    model = model.cuda().eval()

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="LightXML Inference"):
            embeddings = batch["embeddings"].cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32):
                cluster_logits, proj = model(embeddings)
                cluster_probs = torch.sigmoid(cluster_logits)
            for i in range(len(batch["labels"])):
                top_clusters = torch.topk(cluster_probs[i], k=10).indices.cpu().numpy()
                candidates = set()
                for cid in top_clusters:
                    mask = (cluster_assign == cid)
                    candidates.update(np.where(mask)[0].tolist())
                candidates = list(candidates)
                if not candidates:
                    all_preds.append([])
                    continue
                cand_tensor = torch.tensor(candidates, dtype=torch.long, device=proj.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32):
                    cand_emb = model.label_embed(cand_tensor)
                    scores = torch.matmul(cand_emb, proj[i])
                sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
                pred_labels = [idx_to_gnd[candidates[idx]] for idx in sorted_indices]
                all_preds.append(pred_labels[:args.lightxml_top_k])
    return all_preds
def save_embeddings(embeddings, save_path):
    """保存嵌入向量到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"Embeddings saved to {save_path}")
    print(f"Embeddings shape: {embeddings.shape}")
# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightxml_data_dir", type=str, default="./data-test")
    parser.add_argument("--lightxml_model_dir", type=str, default="./arctic-l-ft-head1")
    parser.add_argument("--lightxml_embeddings_path", type=str, default="./arctic-l-ft-emb1/test_embeddings.pt")
    parser.add_argument("--lightxml_csv_file", type=str, default="./data-test/test.csv")
    parser.add_argument("--lightxml_batch_size", type=int, default=512)
    parser.add_argument("--lightxml_hidden_dim", type=int, default=512)
    parser.add_argument("--lightxml_precision", type=str, choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()
    with open(os.path.join(args.lightxml_data_dir, "full_label_map.json")) as f:
        gnd_to_idx = json.load(f)
    label_metadata = load_label_data(gnd_to_idx)
    assert len(label_metadata) == len(gnd_to_idx), f"Label metadata and ground truth labels do not match in length.{len(label_metadata)} != {len(gnd_to_idx)}"
    idx_to_gnd = {v: k for k, v in gnd_to_idx.items()}
    
    # 1. 加载数据
    dev_data = load_dev_data(args.lightxml_csv_file)
    # ground_truths = [item["labels"] for item in dev_data]
    # 从 EmbeddingTestDataset 获取真实标签
    dataset = EmbeddingTestDataset(args.lightxml_embeddings_path, args.lightxml_csv_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ground_truths = []
    for batch in dataloader:
        # 将标签ID转换为GND代码
        labels = [idx_to_gnd[int(idx)] for idx in batch["labels"][0]]  # 修改这一行
        ground_truths.append(labels)

    print("Number of dev examples: ", len(dev_data))
    print("Number of ground truth labels: ",ground_truths[0])
    queries = [item["query"] for item in dev_data]
    


    # 3. LightXML 检索
    print("=== LightXML Retrieval ===")
    

    # 临时替换 args
    args.embeddings_path = args.lightxml_embeddings_path
    args.csv_file = args.lightxml_csv_file
    args.data_dir = args.lightxml_data_dir
    args.model_dir = args.lightxml_model_dir
    args.batch_size = args.lightxml_batch_size
    args.hidden_dim = args.lightxml_hidden_dim
    args.precision = args.lightxml_precision
    args.lightxml_top_k = CONFIG["lightxml_top_k"]

    lightxml_predictions = run_lightxml_inference(args, dev_data, label_metadata, gnd_to_idx, idx_to_gnd)
    print('ligtxml_predictions',lightxml_predictions[0][:10])
    # 2. 向量检索
    print("=== Vector Retrieval ===")
    
    label_texts = [lbl["sentence2"] for lbl in label_metadata]

    model, tokenizer = initialize_model()
    query_embs_path= os.path.join(CONFIG["emb_dir"], "query_embeddings_bf16.pt")
    if os.path.exists(query_embs_path):
        print("Loading existing query embeddings...")
        query_embeddings = torch.load(query_embs_path, weights_only=False).float().numpy()
    else:
        print("Computing query embeddings...")
        query_embeddings = encode_texts(queries, model, tokenizer, CONFIG["query_batch_size"], prefix="query: ")
        save_embeddings(query_embeddings, query_embs_path)
        query_embeddings = query_embeddings.float().numpy()
    # 生成或加载标签嵌入
    label_embeddings_path = os.path.join(CONFIG["emb_dir"], "label_embeddings_bf16.pt")
    if os.path.exists(label_embeddings_path):
        print("Loading existing label embeddings...")
        label_embeddings = torch.load(label_embeddings_path, weights_only=False).float().numpy()
    else:
        print("Computing label embeddings...")
        label_embeddings = encode_texts(label_texts, model, tokenizer, CONFIG["encoding_batch_size"], prefix="")
        save_embeddings(label_embeddings, label_embeddings_path)
        label_embeddings = label_embeddings.float().numpy()

    faiss_index = build_faiss_index(label_embeddings)
    _, indices = faiss_index.search(query_embeddings.astype(np.float32), k=CONFIG["vector_top_k"])

    vector_predictions = []
    for idxs in indices:
        codes = [label_metadata[i]["Code"] for i in idxs if i < len(label_metadata)]
        vector_predictions.append(codes)

    # 4. 融合
    print("=== Fusion (RRF) ===")
    fused_predictions = []
    for vec_pred, xml_pred in zip(vector_predictions, lightxml_predictions):
        fused = reciprocal_rank_fusion(vec_pred[:CONFIG["vector_top_k"]], xml_pred[:CONFIG["lightxml_top_k"]])
        fused_predictions.append(fused[:CONFIG["final_top_k"]])

    # 5. 截断其他结果
    vector_predictions_trunc = [p[:CONFIG["final_top_k"]] for p in vector_predictions]
    lightxml_predictions_trunc = [p[:CONFIG["final_top_k"]] for p in lightxml_predictions]

    # 6. 评估
    ks = [5, 10, 20, 30, 40, 50,100,200,500]
    print("\n" + "="*70)
    print(f"{'Evaluation Results':^70}")
    print("="*70)
    print(f"{'k':<5} {'Method':<12} {'Recall@k':<12} {'Precision@k':<14} {'F1@k':<12}")
    print("-"*70)

    for name, preds in [("Vector", vector_predictions_trunc),
                        ("LightXML", lightxml_predictions_trunc),
                        ("Fused", fused_predictions)]:
        metrics, n_valid = evaluate_metrics(preds, ground_truths, ks=ks)
        for k in ks:
            r = metrics[k]["recall"]
            p = metrics[k]["precision"]
            f1 = metrics[k]["f1"]
            print(f"{k:<5} {name:<12} {r:<12.4f} {p:<14.4f} {f1:<12.4f}")
        print("-"*70)

    print(f"\n✅ Done! Evaluated on {n_valid}/{len(dev_data)} valid samples.")

    #保存fused_predictions为json文件
    with open(os.path.join(CONFIG['emb_dir'],'fused_predictions.json'), 'w') as f:
        json.dump(fused_predictions, f, indent=2, ensure_ascii=False)
    print(f"✅ Done! Saved fused predictions to {os.path.join(CONFIG['emb_dir'],'fused_predictions.json')}.")

    print("Building batched samples (one per query)...")
    batched_samples = []
    for i in tqdm(range(len(dev_data)), desc="Processing queries"):
        gt_set = ground_truths[i]
        cand_codes = fused_predictions[i]# list of 500 indices
        labels_vec = [1 if code in gt_set else 0 for code in cand_codes]
        cand_idxs = [gnd_to_idx[code] for code in cand_codes]
        sample = {
            "query_idx": i,
            "candidate_label_indices": cand_idxs,#索引从0开始
            "labels": labels_vec  # list of 500 binary ints
        }
        batched_samples.append(sample)

    with open(os.path.join('data-super', "rerank_test_all.json"), "w", encoding="utf-8") as f:
        json.dump(batched_samples, f, indent=2)

if __name__ == "__main__":
    main()


