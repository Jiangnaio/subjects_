# prepare_rerank_dataset_batched.py
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
device="cuda"
import pickle
label_cache='/media/jh/新加卷1/2026/rag_code/cache/label_embeddings-s267.pkl'
if os.path.exists(label_cache):
    with open(label_cache, 'rb') as f:
        label_embs, label_meta = pickle.load(f)
    label_embs = torch.from_numpy(label_embs).to(torch.bfloat16)
else:
    label_embs=None


# ================= 配置 =================
CONFIG = {
    #"model_name": "/media/jh/新加卷/11_15/codes/arctic-emb-m-损失函数向量化-b48/final/",
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "lora_path": "/media/jh/新加卷1/2026/rag_code/qwen3-emb-4b-infonce-b16x3-lora48-pos8/checkpoints/checkpoint-267/",
    "max_length": 1500,
    "encoding_batch_size": 32,
    "query_batch_size": 8,
    "search_batch_size":256,
    "retrieval_top_k": 500,
    "output_dir": "./qwen3-rerank_data_lora",
    "dev_file": "/media/jh/新加卷/11_15/codes/datasets/qwen3_embedding_train.json",
    "label_file": "/media/jh/新加卷/11_15/codes/datasets/GND-Subjects-all.json",
    "prefix": "xlm",
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
    

def load_label_data():
    with open(CONFIG["label_file"], 'r', encoding='utf-8') as f:
        gnd_jsons = json.load(f)
    labels = []
    for c in gnd_jsons:
        alname = ','.join(str(x) for x in c.get("Alternate Name", []) or []) or "None"
        rs = ','.join(str(x) for x in c.get("Related Subjects", []) or []) or "None"
        definition = c.get("Definition", "None") or "None"
        classification = c.get("Classification Name", "Unknown")
        name = c.get("Name", "Unknown")
        passage_text = f"Classification: {classification}. Subject: {name}. Alternate Names: {alname}. Related: {rs}. Definition: {definition}"
        item = {
            'sentence2': passage_text,
            'Code': str(c['Code']),
        }
        labels.append(item)
    return labels

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
def encode_texts(texts, model, tokenizer, batch_size=32, prefix=""):
    model.eval()
    device = next(model.parameters()).device
    all_embeddings = []
    texts = [f"{prefix}{t}" for t in texts]
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        ).to(device)
        embeddings = last_token_pool(model(**inputs).last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1).to(torch.bfloat16)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

@torch.no_grad()
def search_topk_gpu(query_embs, label_embs, top_k=500, batch_size=32):
    device = label_embs.device
    label_embs = label_embs.to(device)
    N = query_embs.size(0)
    all_indices = []
    for i in tqdm(range(0, N, batch_size), desc="GPU Similarity Search"):
        q_batch = query_embs[i:i+batch_size].to(device)
        sim = torch.mm(q_batch, label_embs.t())
        _, topk_idx = torch.topk(sim, k=min(top_k, sim.size(1)), dim=1, largest=True)
        all_indices.append(topk_idx.cpu())
    return torch.cat(all_indices, dim=0)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_debug", action="store_true", 
                        help="test with a small subset of data")
    args = parser.parse_args()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print("Initializing model...")
    model, tokenizer = initialize_model()

    print("Loading labels...")
    labels = load_label_data()
    label_codes = [lbl['Code'] for lbl in labels]
    label_texts = [lbl['sentence2'] for lbl in labels]
    if label_embs != None:
        label_embeddings=label_embs
        print(f"从{label_cache}加载标签编码向量")
    else:
        print("Encoding labels...")
        label_embeddings = encode_texts(label_texts, model, tokenizer, CONFIG["encoding_batch_size"], "passage: ")
    torch.save(label_embeddings, os.path.join(CONFIG["output_dir"], "label_embeddings.pt"))
    with open(os.path.join(CONFIG["output_dir"], "label_codes.json"), "w", encoding="utf-8") as f:
        json.dump(label_codes, f, ensure_ascii=False, indent=2)

    print("Loading dev data...")
    dev_data = load_dev_data(CONFIG["dev_file"])
    if args.use_debug:
        dev_data = dev_data[:100]
    queries = [f"Instruct: {TASK_DESCRIPTION}\nQuery: {item['query']}" for item in dev_data]
    ground_truths = [set(item['positive_gndids']) for item in dev_data]

    print("Encoding queries...")
    query_embeddings = encode_texts(queries, model, tokenizer, CONFIG["query_batch_size"], "")
    torch.save(query_embeddings, os.path.join(CONFIG["output_dir"], "query_embeddings.pt"))

    print("Searching top-500 candidates...")
    candidate_indices = search_topk_gpu(
        query_embs=query_embeddings,
        label_embs=label_embeddings,
        top_k=CONFIG["retrieval_top_k"],
        batch_size=CONFIG["search_batch_size"]
    )  # (N, 500)

    print("Building batched samples (one per query)...")
    batched_samples = []
    for i in tqdm(range(len(dev_data)), desc="Processing queries"):
        gt_set = ground_truths[i]
        cand_idxs = candidate_indices[i].tolist()  # list of 500 indices
        labels_vec = [1 if label_codes[idx] in gt_set else 0 for idx in cand_idxs]
        sample = {
            "query_idx": i,
            "candidate_label_indices": cand_idxs,
            "labels": labels_vec  # list of 500 binary ints
        }
        batched_samples.append(sample)

    with open(os.path.join(CONFIG["output_dir"], "rerank_train_samples_batched.json"), "w", encoding="utf-8") as f:
        json.dump(batched_samples, f, indent=2)

    print(f"✅ Saved {len(batched_samples)} batched samples.")
    print(f"✅ Embeddings saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
