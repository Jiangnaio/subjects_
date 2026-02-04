import os
import random
import math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import gc
import time
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
from contextlib import nullcontext
import pandas as pd

outdir = './arctic-m-info_nce-b48-lora48-pos7'

# 配置参数（全参数微调，使用 Qwen3 官方 InfoNCE 损失）
CONFIG = {
    "seed": 32,
    "model_name": "Snowflake/snowflake-arctic-embed-m-v2.0",
    "max_length": 2048,
    "batch_size": 48,# m设置为32*1; l设置为8*4
    "gradient_accumulation_steps": 1,
    "num_epochs": 2,
    "learning_rate": 5e-5,
    "lora_r": 48,           # 降低LoRA秩 0.6B为32, 4B为24
    "lora_alpha": 96,
    "lora_dropout": 0.05,
    "warmup_ratio": 0.1,
    "output_dir": outdir,
    "dataset_dir": "./datasets2",
    "train_file": "./datasets2/qwen3_embedding_train_dedup.json",
    "eval_file": "qwen3_embedding_dev_dedup.json",
    "debug_mode": False,
    "debug_size": 50,
    "eval_steps": 300,
    "save_steps": 300,
    "save_plot_steps": 300,
    "max_keep_checkpoints": 20,
    "logging_steps": 300,
    "memory_cleanup_steps": 50,
    "max_memory_percent": 85,
    "resume_from_checkpoint": False,
    "checkpoint_dir": f"{outdir}/checkpoints",
    "log_file": f"{outdir}/training_log.json",
    "plot_dir": f"{outdir}/plots",
    "use_gradient_checkpointing": True,
    "use_flash_attention": True,
    "bf16": True, # 使用fp32
    "max_negatives": 10,                   # 增加负样本数量，通过增加负样本的数量，却是能够帮助模型更好地学习到不同样本之间的差异，从而提高模型的性能。
    "max_positives": 7,   #8,3 --> 10,5 --> 10,7 --> 10,50(不限制正主题数量)
    "max_hard_negatives": 0, # 硬负样本
    "temperature": 0.05,  # Qwen3 官方温度值0.05
    "early_stopping_patience": 10, #训练到step 2800触发早停（约0.55 epoch）
    "min_loss_change": 1e-4,
    "use_dynamic_lr": True,
    "eval_batch_size_multiplier": 1,
    "clip_initial_value": 1.0,
    "clip_final_value": 0.3,
    "loss_type": "info_nce",  # 可选"info_nce"或"supcon"
    "use_dynamic_temperature": True,
    "temp_initial": 0.05,
    "temp_max": 0.15, # 温度最大值
    "temp_final": 0.08,
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ All random seeds fixed to {seed}")

TASK_DESCRIPTION = "Given a paper's title and abstract, retrieve relevant subject topics"

def get_dynamic_temperature(global_step, total_steps): #固定温度
    return 0.05
  
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        used_percent = (allocated / total) * 100
        return allocated, used_percent
    return 0, 0

def dedup_preserve_order(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer, max_length=8192, max_positives=10, max_negatives=5, debug_mode=False, debug_size=100, max_hard_negatives=2):
        with open(data_file, 'r', encoding='utf-8') as f:
            if debug_mode:
                self.examples = json.load(f)[:debug_size]
            else:
                self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        print(f"Loaded {len(self.examples)} samples from {data_file}")

    def __len__(self):
        return len(self.examples)

    def _get_single_item(self, idx):
        item = self.examples[idx]
        query = item["query"]
        if 'Instruct' not in query:
            query = f"Instruct: {TASK_DESCRIPTION}\nQuery: {query}"
        positive_topics = dedup_preserve_order(item["positive_topics"])[:self.max_positives]
        negative_topics = item["negative_topics"][:self.max_negatives]
        max_hard_negatives = min(self.max_hard_negatives, len(positive_topics))
        hard_negative_topics = item.get("hard_negative_topics", [])[-max_hard_negatives:]
        return {
            "query": query,
            "positive_topics": positive_topics,
            "negative_topics": negative_topics,
            "hard_negative_topics": hard_negative_topics,
            "doc_id": item["metadata"]["doc_id"]
        }

    def __getitem__(self, idx):
        return self._get_single_item(idx)

class ContrastiveDataCollator:
    def __init__(self, tokenizer, max_length=8192, max_positives=5, max_negatives=10, max_hard_negatives=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.K = max_positives + max_negatives + max_hard_negatives
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives

    def __call__(self, features):
        queries = [f["query"] for f in features]
        B = len(queries)
        K = self.K
        
        # 构建 texts: [B, K], labels: [B, K]
        all_texts = []
        labels = []
        
        for f in features:
            # 收集所有候选
            candidates = []
            label_list = []
            
            # 正样本
            pos = f["positive_topics"][:self.max_positives]
            candidates.extend(pos)
            label_list.extend([1] * len(pos))
            
            # 负样本
            neg = f["negative_topics"][:self.max_negatives]
            candidates.extend(neg)
            label_list.extend([0] * len(neg))
            
            # 硬负样本
            hard_neg = f.get("hard_negative_topics", [])[:self.max_hard_negatives]
            candidates.extend(hard_neg)
            label_list.extend([0] * len(hard_neg))
            
            # 补齐到 K（用最后一个元素重复，或空字符串）
            if len(candidates) < K:
                pad_len = K - len(candidates)
                pad_token = candidates[-1] if candidates else ""  # 避免空
                candidates.extend([pad_token] * pad_len)
                label_list.extend([0] * pad_len)  # padding 视为负样本（但 loss 中会 mask？）
            elif len(candidates) > K:
                candidates = candidates[:K]
                label_list = label_list[:K]
            
            all_texts.append(candidates)
            labels.append(label_list)
        
        # Tokenize queries: [B, L]
        query_encodings = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize texts: flatten to [B*K, L], then reshape
        flat_texts = [text for row in all_texts for text in row]  # [B*K]
        text_encodings = self.tokenizer(
            flat_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        # Reshape input_ids and attention_mask to [B, K, L]
        L = text_encodings["input_ids"].size(1)
        text_input_ids = text_encodings["input_ids"].view(B, K, L)
        text_attention_mask = text_encodings["attention_mask"].view(B, K, L)
        
        labels = torch.tensor(labels, dtype=torch.long)  # [B, K]
        
        return {
            "query_input_ids": query_encodings["input_ids"],
            "query_attention_mask": query_encodings["attention_mask"],
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "labels": labels,  # [B, K]
            "num_queries": B
        }
def load_datasets(tokenizer):
    train_path = CONFIG["train_file"]
    eval_path = os.path.join(CONFIG["dataset_dir"], CONFIG["eval_file"])
    train_dataset = ContrastiveDataset(
        train_path, tokenizer,
        max_length=CONFIG["max_length"],
        max_positives=CONFIG["max_positives"],
        max_negatives=CONFIG["max_negatives"],
        debug_mode=CONFIG["debug_mode"],
        debug_size=CONFIG["debug_size"],
        max_hard_negatives=CONFIG["max_hard_negatives"]
    )
    eval_dataset = ContrastiveDataset(
        eval_path, tokenizer,
        max_length=CONFIG["max_length"],
        max_positives=CONFIG["max_positives"],
        max_negatives=CONFIG["max_negatives"],
        debug_mode=CONFIG["debug_mode"],
        debug_size=CONFIG["debug_size"],
        max_hard_negatives=CONFIG["max_hard_negatives"]
    )
    eval_indices = random.sample(range(len(eval_dataset)), min(200, len(eval_dataset)))
    eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)
    return train_dataset, eval_dataset

def initialize_model():
    from transformers import AutoConfig
    from peft import get_peft_model, LoraConfig, TaskType
    config = AutoConfig.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    print(f"🔧 Model config hidden_size: {config.hidden_size}")  # 确保是 768

    if hasattr(config, "use_memory_efficient_attention"):
        config.use_memory_efficient_attention = False

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    dtype = torch.bfloat16 if CONFIG["bf16"] and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModel.from_pretrained(
        CONFIG["model_name"],
        config=config,
        add_pooling_layer=False,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # 启用梯度检查点
    if CONFIG["use_gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("✅ Gradient checkpointing enabled (reduces memory by ~40%)")
    
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "up_gate_proj", "down_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 移动到GPU
    print("Moving model to GPU...")
    model = model.to("cuda")
    
    # 打印参数信息
    trainable_params, total_params, ratio = count_trainable_parameters(model)
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({ratio:.2f}%)")
    print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"{'='*50}")
    
    return model, tokenizer, dtype

def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params, trainable_params / total_params * 100

def get_embeddings(model, input_ids, attention_mask, dtype=None):
    """
    Handles both [B, L] and [B, K, L] inputs
    """
    original_shape = input_ids.shape
    if len(original_shape) == 3:
        B, K, L = original_shape
        # Flatten to [B*K, L]
        input_ids = input_ids.view(B * K, L)
        attention_mask = attention_mask.view(B * K, L)
        is_text_batch = True
    else:
        is_text_batch = False

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0]  # [N, D]

    if is_text_batch:
        embeddings = embeddings.view(B, K, -1)  # [B, K, D]

    del outputs.last_hidden_state, outputs
    torch.cuda.empty_cache()
    return F.normalize(embeddings, p=2, dim=-1)

def compute_multi_positive_info_nce_loss(query_embeds, text_embeds, labels, temperature=0.05):
    """
    向量化 Multi-Positive InfoNCE Loss
    Args:
        query_embeds: [B, D]
        text_embeds: [B, K, D]
        labels: [B, K]  (1=positive, 0=negative)
        temperature: scalar
    Returns:
        scalar loss
    """
    B, K, D = text_embeds.shape
    
    # Compute cosine similarities: [B, K]
    sim = torch.einsum('bd,bkd->bk', query_embeds, text_embeds) / temperature
    
    # Numerator: log-sum-exp of positive samples for each query
    # Mask: [B, K]
    pos_mask = (labels == 1)  # [B, K]
    pos_sim = sim.masked_fill(~pos_mask, -1e9)  # 非正样本设为极小值
    numerator = torch.logsumexp(pos_sim, dim=1)    # [B]
    
    # Denominator: log-sum-exp over all K candidates
    denominator = torch.logsumexp(sim, dim=1)      # [B]
    
    # Loss per query: [B]
    loss_per_query = denominator - numerator       # [B]
    
    # Remove queries with no positive samples (though should not happen)
    valid_mask = pos_mask.any(dim=1)  # [B]
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=query_embeds.device, requires_grad=True)
    
    loss = loss_per_query[valid_mask].mean()
    return loss


# def compute_multi_positive_info_nce_loss(query_embeds, text_embeds, labels, temperature=0.05):
#     """
#     向量化 Multi-Positive InfoNCE Loss (带样本加权)
#     Args:
#         query_embeds: [B, D]
#         text_embeds: [B, K, D]
#         labels: [B, K]  (1=positive, 0=negative)
#         temperature: scalar
#     Returns:
#         scalar loss
#     """
#     B, K, D = text_embeds.shape
    
#     # Compute cosine similarities: [B, K]
#     sim = torch.einsum('bd,bkd->bk', query_embeds, text_embeds) / temperature
    
#     # Numerator: log-sum-exp of positive samples for each query
#     pos_mask = (labels == 1)  # [B, K]
#     pos_sim = sim.masked_fill(~pos_mask, -1e9)  # 非正样本设为极小值
#     numerator = torch.logsumexp(pos_sim, dim=1)    # [B]
    
#     # Denominator: log-sum-exp over all K candidates
#     denominator = torch.logsumexp(sim, dim=1)      # [B]
    
#     # Loss per query: [B]
#     loss_per_query = denominator - numerator       # [B]
    
#     # Remove queries with no positive samples (though should not happen)
#     valid_mask = pos_mask.any(dim=1)  # [B]
#     if valid_mask.sum() == 0:
#         return torch.tensor(0.0, device=query_embeds.device, requires_grad=True)
    
#     # --- 动态加权：倒数加权 ---
#     # 计算每个样本的正样本数量
#     num_positives = pos_mask.sum(dim=1)  # [B]
#     # 使用倒数作为权重，避免除零，添加一个小常数 epsilon
#     epsilon = 1#e-6
#     weights = 1.0 / (num_positives + epsilon)  # [B]
#     # 对 loss_per_query 进行加权
#     weighted_loss_per_query = loss_per_query * weights
#     # 取加权平均
#     loss = weighted_loss_per_query[valid_mask].mean()
#     # --- 结束加权 ---
    
#     return loss

def compute_supcon_style_loss(query_embeds, text_embeds, labels, temperature=0.05):
    """
    Batch-wise Supervised Contrastive Loss (Query-Anchored)
    
    Args:
        query_embeds: [B, D]
        text_embeds: [B, K, D]
        labels: [B, K] (1=positive, 0=negative)
        temperature: float
    
    Returns:
        scalar loss
    """
    device = query_embeds.device
    B, K, D = text_embeds.shape

    # Flatten text embeddings: [B*K, D]
    text_embeds_flat = text_embeds.view(-1, D)  # [B*K, D]
    labels_flat = labels.view(-1)  # [B*K]

    # Compute similarity matrix: [B, B*K]
    # sim[i, j] = cosine similarity between query i and topic j
    sim_matrix = torch.matmul(query_embeds, text_embeds_flat.T) / temperature  # [B, B*K]

    # Build positive mask: [B, B*K]
    # Only topics from the same query-i can be positive for query-i
    pos_mask = torch.zeros(B, B*K, dtype=torch.bool, device=device)
    for i in range(B):
        start = i * K
        end = (i + 1) * K
        pos_mask[i, start:end] = (labels[i] == 1)

    # Remove queries with no positives
    valid_query_mask = pos_mask.any(dim=1)  # [B]
    if not valid_query_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Numerator: log-sum-exp of positives for each valid query
    sim_pos = sim_matrix.masked_fill(~pos_mask, -1e9)  # [B, B*K]
    numerator = torch.logsumexp(sim_pos, dim=1)  # [B]

    # Denominator: log-sum-exp over ALL candidates (B*K)
    denominator = torch.logsumexp(sim_matrix, dim=1)  # [B]

    # Loss per query
    loss_per_query = denominator - numerator  # [B]

    # Mean over valid queries
    loss = loss_per_query[valid_query_mask].mean()
    return loss

def evaluate(model, dataloader, device, dtype, temperature=0.05):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            query_input_ids = batch["query_input_ids"].to(device, non_blocking=True)
            query_attention_mask = batch["query_attention_mask"].to(device, non_blocking=True)
            text_input_ids = batch["text_input_ids"].to(device, non_blocking=True)
            text_attention_mask = batch["text_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            num_queries = batch["num_queries"]
            try:
                query_embeds = get_embeddings(model, query_input_ids, query_attention_mask, dtype)
                text_embeds = get_embeddings(model, text_input_ids, text_attention_mask, dtype)
                if CONFIG["loss_type"] == "info_nce":
                    loss = compute_multi_positive_info_nce_loss(
                        query_embeds, text_embeds, labels,
                        temperature=temperature
                    )
                elif CONFIG["loss_type"] == "supcon":
                    loss = compute_supcon_style_loss(
                        query_embeds, text_embeds, labels,
                        temperature=temperature
                    )
                total_loss += loss.item() * num_queries
                total_samples += num_queries
            finally:
                del query_input_ids, query_attention_mask, text_input_ids, text_attention_mask
                del labels, query_embeds, text_embeds, loss
                torch.cuda.empty_cache()
                gc.collect()
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    cleanup_memory()
    model.train()
    return avg_loss

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
    time.sleep(0.01)

def manage_checkpoints(output_dir, max_keep):
    if not os.path.exists(output_dir):
        return
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if len(checkpoints) > max_keep:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        for checkpoint in checkpoints[:len(checkpoints) - max_keep]:
            path = os.path.join(output_dir, checkpoint)
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error removing {checkpoint}: {e}")

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])

def load_training_state(checkpoint_path):
    state_file = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(state_file):
        return torch.load(state_file, weights_only=False)
    return None

def save_training_state(checkpoint_path, state):
    state_file = os.path.join(checkpoint_path, "training_state.pt")
    torch.save(state, state_file)

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_eval_loss,
                   training_logs, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
    save_model(model, tokenizer, checkpoint_path)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "training_logs": training_logs,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    save_training_state(checkpoint_path, state)
    manage_checkpoints(checkpoint_dir, CONFIG["max_keep_checkpoints"])
    print(f"✓ Checkpoint saved at step {global_step}")

def create_training_plots(training_logs, plot_dir):
    if not training_logs:
        return
    os.makedirs(plot_dir, exist_ok=True)
    steps = np.array([log['step'] for log in training_logs])
    train_losses = np.array([log['train_loss'] for log in training_logs])
    eval_losses = []
    eval_steps = []
    for log in training_logs:
        if 'eval_loss' in log and log['eval_loss'] is not None and not np.isnan(log['eval_loss']):
            eval_losses.append(log['eval_loss'])
            eval_steps.append(log['step'])
    eval_losses = np.array(eval_losses)
    eval_steps = np.array(eval_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', label='Train Loss')
    if len(eval_losses) > 0:
        plt.plot(eval_steps, eval_losses, 'ro-', label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"loss_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
    plt.close()

def adjust_training_dynamics(optimizer, global_step, total_steps):
    if CONFIG["use_dynamic_lr"]:
        if global_step < total_steps * 0.1:
            lr_factor = 0.5
        elif global_step > total_steps * 0.8:
            lr_factor = 0.2
        else:
            lr_factor = 1.0
        for pg in optimizer.param_groups:
            pg['lr'] = CONFIG["learning_rate"] * lr_factor
    progress = global_step / max(total_steps, 1)
    clip_value = CONFIG["clip_initial_value"] - (CONFIG["clip_initial_value"] - CONFIG["clip_final_value"]) * progress
    return clip_value

# 主训练流程
if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    for d in [CONFIG["output_dir"], CONFIG["checkpoint_dir"], CONFIG["plot_dir"]]:
        os.makedirs(d, exist_ok=True)

    print(f"Starting FULL FINE-TUNING of Arcitc Embedding with InfoNCE loss...")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])


    train_dataset, eval_dataset = load_datasets(tokenizer)
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    collator = ContrastiveDataCollator(
        tokenizer, 
        max_length=CONFIG["max_length"],
        max_positives=CONFIG["max_positives"],
        max_negatives=CONFIG["max_negatives"],
        max_hard_negatives=CONFIG["max_hard_negatives"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collator, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"] * CONFIG["eval_batch_size_multiplier"], shuffle=False, collate_fn=collator, pin_memory=True)

    total_steps = len(train_dataloader) // CONFIG["gradient_accumulation_steps"] * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    start_epoch, global_step, best_eval_loss = 0, 0, float('inf')
    training_logs = []
    early_stop_counter = 0
    model = None

    if CONFIG["resume_from_checkpoint"]:
        latest_checkpoint = find_latest_checkpoint(CONFIG["checkpoint_dir"])
        if latest_checkpoint:
            print(f"🔍 Resuming from checkpoint: {latest_checkpoint}")

            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
            if CONFIG["bf16"]:
                dtype = torch.bfloat16
                model = AutoModel.from_pretrained(
                    CONFIG["model_name"], 
                    add_pooling_layer=False,
                    dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    # attn_implementation="flash_attention_2"  # 新增 flash-attn还不支持Bert模型
                )
            else:
                dtype = torch.float32
                model = AutoModel.from_pretrained(
                    CONFIG["model_name"], 
                    add_pooling_layer=False,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    )
            model.to(device)
            if CONFIG["use_gradient_checkpointing"]:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("✅ Gradient checkpointing enabled")

            state = load_training_state(latest_checkpoint)
            if state:
                start_epoch = state["epoch"]
                global_step = state["global_step"]
                best_eval_loss = state["best_eval_loss"]
                training_logs = state["training_logs"]
                random.setstate(state["random_state"])
                np.random.set_state(state["numpy_random_state"])
                torch.set_rng_state(state["torch_random_state"])
                if state["torch_cuda_random_state"] is not None:
                    torch.cuda.set_rng_state(state["torch_cuda_random_state"])
                print(f"Resuming from epoch {start_epoch}, step {global_step}")

    if model is None:
        model, tokenizer, dtype = initialize_model()

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if CONFIG["resume_from_checkpoint"] and 'latest_checkpoint' in locals() and latest_checkpoint:
        opt_path = os.path.join(latest_checkpoint, "optimizer.pt")
        sch_path = os.path.join(latest_checkpoint, "scheduler.pt")
        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            scheduler.load_state_dict(torch.load(sch_path, map_location=device))
            print("✅ Optimizer and scheduler restored")

    batches_to_skip = global_step * CONFIG["gradient_accumulation_steps"] if start_epoch == 0 else 0
    best_model_path = os.path.join(CONFIG["output_dir"], "best_model")

    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    temp = get_dynamic_temperature(global_step, total_steps)
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        accumulated_loss, num_acc_steps = 0.0, 0

        batch_iter = enumerate(progress_bar)
        if epoch == start_epoch and batches_to_skip > 0:
            print(f"⏩ Skipping {batches_to_skip} batches...")
            for _ in range(batches_to_skip):
                try:
                    next(batch_iter)
                except StopIteration:
                    break

        for step, batch in batch_iter:
            
            query_input_ids = batch["query_input_ids"].to(device, non_blocking=True)
            query_attention_mask = batch["query_attention_mask"].to(device, non_blocking=True)
            text_input_ids = batch["text_input_ids"].to(device, non_blocking=True)
            text_attention_mask = batch["text_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            temp = get_dynamic_temperature(global_step, total_steps)
            try:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    q_emb = get_embeddings(model, query_input_ids, query_attention_mask, dtype)
                    t_emb = get_embeddings(model, text_input_ids, text_attention_mask, dtype)
                    if CONFIG["loss_type"]=="info_nce":
                        loss = compute_multi_positive_info_nce_loss(
                            q_emb, t_emb, labels,
                            temperature=temp
                        )
                    elif CONFIG["loss_type"] == "supcon":
                        loss = compute_supcon_style_loss(
                            q_emb, t_emb, labels,
                            temperature=temp
                        )

            

                scaled_loss = loss / CONFIG["gradient_accumulation_steps"]
                scaled_loss.backward()

                if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0 or (step + 1) == len(train_dataloader):
                    clip_value = adjust_training_dynamics(optimizer, global_step, total_steps)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    avg_loss = loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    allocated, _ = get_gpu_memory_usage()
                    log_entry = {
                        "step": global_step + 1,
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "learning_rate": current_lr,
                        "gpu_memory": allocated,
                        "timestamp": datetime.now().isoformat()
                    }
                    training_logs.append(log_entry)

                    global_step += 1

                    if global_step % CONFIG["logging_steps"] == 0:
                        print(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | GPU: {allocated:.2f}GB")

                    if global_step % CONFIG["eval_steps"] == 0:
                        eval_loss = evaluate(model, eval_dataloader, device, dtype, temp)
                        log_entry['eval_loss'] = eval_loss
                        if eval_loss < best_eval_loss - CONFIG["min_loss_change"]:
                            best_eval_loss = eval_loss
                            save_model(model, tokenizer, best_model_path)
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= CONFIG["early_stopping_patience"]:
                                print("Early stopping triggered.")
                                save_model(model, tokenizer, os.path.join(CONFIG["output_dir"], "final_early_stop"))
                                exit(0)

                        save_checkpoint(
                            model, optimizer, scheduler, epoch, global_step, best_eval_loss,
                            training_logs, CONFIG["checkpoint_dir"]
                        )
                    if global_step % CONFIG["save_plot_steps"] == 0:
                        create_training_plots(training_logs, CONFIG["plot_dir"])

            finally:
                del query_input_ids, query_attention_mask, text_input_ids, text_attention_mask, labels
                if 'q_emb' in locals(): del q_emb
                if 't_emb' in locals(): del t_emb
                if 'loss' in locals(): del loss
                cleanup_memory()

        final_eval_loss = evaluate(model, eval_dataloader, device, dtype, temp)
        print(f"Epoch {epoch+1} finished. Final eval loss: {final_eval_loss:.4f}")
        save_model(model, tokenizer, os.path.join(CONFIG["output_dir"], f"epoch-{epoch+1}"))

    save_model(model, tokenizer, os.path.join(CONFIG["output_dir"], "final"))
    with open(CONFIG["log_file"], 'w') as f:
        json.dump(training_logs, f, indent=2)
    create_training_plots(training_logs, CONFIG["plot_dir"])
    print("✅ Full fine-tuning with InfoNCE loss completed.")
