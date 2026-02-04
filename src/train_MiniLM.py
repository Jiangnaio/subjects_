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

outdir = 'mpnet-base-v2-b8x4'#'./MiniLM-train-b8x4'  # 修改输出目录名

# 配置参数（适配 MiniLM）
CONFIG = {
    "seed": 42,
    "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",#"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # ✅ 替换为 MiniLM
    "max_length": 512,  # ✅ MiniLM 最大支持 512，但 128 足够，节省显存
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "learning_rate": 2e-5,  # ✅ MiniLM 常用 2e-5，比 arctic 的 1e-5 更高
    "warmup_ratio": 0.1,
    "output_dir": outdir,
    "dataset_dir": "./datasets1",
    "train_file": "qwen3_embedding_train.json",
    "eval_file": "qwen3_embedding_dev.json",
    "debug_mode": False,
    "debug_size": 50,
    "eval_steps": 100,
    "save_steps": 100,
    "save_plot_steps": 100,
    "max_keep_checkpoints": 20,
    "logging_steps": 300,
    "memory_cleanup_steps": 50,
    "max_memory_percent": 85,
    "resume_from_checkpoint": False,
    "checkpoint_dir": f"{outdir}/checkpoints",
    "log_file": f"{outdir}/training_log.json",
    "plot_dir": f"{outdir}/plots",
    "use_gradient_checkpointing": False,  # ✅ MiniLM 不推荐，可能反而慢
    "use_flash_attention": False,         # ✅ MiniLM 不支持 flash attention
    "bf16": True,                       
    "max_negatives": 10,
    "max_positives": 5,
    "max_hard_negatives": 0,
    "temperature": 0.05,
    "early_stopping_patience": 5,
    "min_loss_change": 1e-4,
    "use_dynamic_lr": True,
    "eval_batch_size_multiplier": 1,
    "clip_initial_value": 1.0,
    "clip_final_value": 0.3,
    "loss_type": "info_nce",
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
        
        all_texts = []
        labels = []
        
        for f in features:
            candidates = []
            label_list = []
            
            pos = f["positive_topics"][:self.max_positives]
            candidates.extend(pos)
            label_list.extend([1] * len(pos))
            
            neg = f["negative_topics"][:self.max_negatives]
            candidates.extend(neg)
            label_list.extend([0] * len(neg))
            
            hard_neg = f.get("hard_negative_topics", [])[:self.max_hard_negatives]
            candidates.extend(hard_neg)
            label_list.extend([0] * len(hard_neg))
            
            if len(candidates) < K:
                pad_len = K - len(candidates)
                pad_token = candidates[-1] if candidates else ""
                candidates.extend([pad_token] * pad_len)
                label_list.extend([0] * pad_len)
            elif len(candidates) > K:
                candidates = candidates[:K]
                label_list = label_list[:K]
            
            all_texts.append(candidates)
            labels.append(label_list)
        
        query_encodings = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        flat_texts = [text for row in all_texts for text in row]
        text_encodings = self.tokenizer(
            flat_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        L = text_encodings["input_ids"].size(1)
        text_input_ids = text_encodings["input_ids"].view(B, K, L)
        text_attention_mask = text_encodings["attention_mask"].view(B, K, L)
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            "query_input_ids": query_encodings["input_ids"],
            "query_attention_mask": query_encodings["attention_mask"],
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "labels": labels,
            "num_queries": B
        }

def load_datasets(tokenizer):
    train_path = os.path.join(CONFIG["dataset_dir"], CONFIG["train_file"])
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
    # ✅ MiniLM 是标准 HuggingFace 模型，无需 trust_remote_code 或特殊 config
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # ✅ 使用 bf16（如果支持），否则 fp32
    dtype = torch.bfloat16 if CONFIG["bf16"] else torch.float32

    model = AutoModel.from_pretrained(
        CONFIG["model_name"],
        # ✅ 不要加 add_pooling_layer=False！MiniLM 有默认 pooler
        # ✅ 移除 trust_remote_code, low_cpu_mem_usage
        torch_dtype=dtype,
    )

    if CONFIG["use_gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")

    model = model.to("cuda")
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

def mean_pooling(model_output, attention_mask):
    """
    ✅ 使用 MiniLM 官方推荐的 mean pooling，而非 [CLS]
    参考：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
    """
    token_embeddings = model_output[0]  # First element is token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(model, input_ids, attention_mask, dtype=None):
    """
    支持 [B, L] 和 [B, K, L] 输入，使用 mean_pooling
    """
    original_shape = input_ids.shape
    if len(original_shape) == 3:
        B, K, L = original_shape
        input_ids = input_ids.view(B * K, L)
        attention_mask = attention_mask.view(B * K, L)
        is_text_batch = True
    else:
        is_text_batch = False

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    
    embeddings = mean_pooling(outputs, attention_mask)  # ✅ 使用 mean pooling
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    if is_text_batch:
        embeddings = embeddings.view(B, K, -1)

    del outputs
    torch.cuda.empty_cache()
    return embeddings

def compute_multi_positive_info_nce_loss(query_embeds, text_embeds, labels, temperature=0.05):
    """
    向量化 Multi-Positive InfoNCE Loss —— 无需修改，完全兼容
    """
    B, K, D = text_embeds.shape
    
    sim = torch.einsum('bd,bkd->bk', query_embeds, text_embeds) / temperature
    pos_mask = (labels == 1)
    pos_sim = sim.masked_fill(~pos_mask, -1e4)
    numerator = torch.logsumexp(pos_sim, dim=1)
    denominator = torch.logsumexp(sim, dim=1)
    loss_per_query = denominator - numerator
    
    valid_mask = pos_mask.any(dim=1)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=query_embeds.device, requires_grad=True)
    
    loss = loss_per_query[valid_mask].mean()
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
                        temperature=CONFIG["temperature"]
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

    print(f"Starting FULL FINE-TUNING of MiniLM with InfoNCE loss...")
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
            dtype = torch.bfloat16 if CONFIG["bf16"] else torch.float32
            model = AutoModel.from_pretrained(latest_checkpoint, torch_dtype=dtype)
            if CONFIG["use_gradient_checkpointing"]:
                model.gradient_checkpointing_enable()
            model.to(device)
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

            try:
                # ✅ 使用自动混合精度（bf16）
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if CONFIG["bf16"] else torch.float32):
                    q_emb = get_embeddings(model, query_input_ids, query_attention_mask, dtype)
                    t_emb = get_embeddings(model, text_input_ids, text_attention_mask, dtype)
                    if CONFIG["loss_type"]=="info_nce":
                        loss = compute_multi_positive_info_nce_loss(
                            q_emb, t_emb, labels,
                            temperature=CONFIG["temperature"]
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
                        eval_loss = evaluate(model, eval_dataloader, device, dtype, CONFIG["temperature"])
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

        final_eval_loss = evaluate(model, eval_dataloader, device, dtype, CONFIG["temperature"])
        print(f"Epoch {epoch+1} finished. Final eval loss: {final_eval_loss:.4f}")
        save_model(model, tokenizer, os.path.join(CONFIG["output_dir"], f"epoch-{epoch+1}"))

    save_model(model, tokenizer, os.path.join(CONFIG["output_dir"], "final"))
    with open(CONFIG["log_file"], 'w') as f:
        json.dump(training_logs, f, indent=2)
    create_training_plots(training_logs, CONFIG["plot_dir"])
    print("✅ Full fine-tuning with InfoNCE loss completed.")