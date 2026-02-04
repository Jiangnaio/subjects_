# 1_precompute_embeddings.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np
MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'
model_dim = 1024
expand_factor = 6

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1500):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        inputs = self.tokenizer(
            text, max_length=self.max_length, padding=True,
            truncation=True, return_tensors="pt"
        )#padding="max_length"固定为按照最大长度编码，耗时较长。
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

# def collate_fn(batch):# 该处理函数只能用于按照固定长度截断或填充的输入
#     input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
#     attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
#     return {"input_ids": input_ids, "attention_mask": attention_mask}
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def collate_fn(batch):#该处理函数可以保证batch内输入长度相同，batch之间的输入长度可以不同，大幅度节省编码时间
    # 提取 input_ids 和 attention_mask 列表（都是 list of ints）
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        max_length=1024,
        return_tensors="pt"
    )
    return padded
class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.roberta = AutoModel.from_pretrained(
            MODEL_NAME,
            add_pooling_layer=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        # We only need the concatenated CLS from last 4 layers

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embs = torch.cat([outputs.hidden_states[-i][:, 0] for i in range(1, 1+expand_factor)], dim=-1)
            return cls_embs  # [B, 4096]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--use_test", type=str, default=None) 
    parser.add_argument("--model_name", type=str, default='Snowflake/snowflake-arctic-embed-l-v2.0') 
    parser.add_argument("--output_dir", type=str, default="./embeddings")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--expand_factor", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global expand_factor
    expand_factor = args.expand_factor
    print('expand_factor',expand_factor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_bf16_supported(), "BF16 required"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.use_test:
        dataset = TextDataset(os.path.join(args.use_test, "test.csv"), tokenizer)
    else:
        dataset = TextDataset(os.path.join(args.data_dir, "train.csv"), tokenizer)
    print('数据集数量',len(dataset))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    model = FrozenEncoder().to(device).eval()

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Precomputing embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embs = model(input_ids, attention_mask)  # [B, 4096]
            all_embeddings.append(embs.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, 4096]
    if args.use_test:
        torch.save(embeddings, os.path.join(args.output_dir, "test_embeddings.pt"))
    else:
        torch.save(embeddings, os.path.join(args.output_dir, "train_embeddings.pt"))
    print(f"✅ Saved embeddings shape: {embeddings.shape} to {args.output_dir}")

if __name__ == "__main__":
    main()