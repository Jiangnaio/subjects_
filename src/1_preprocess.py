# 1_preprocess.py
import os
import json
import pandas as pd
from collections import defaultdict
import argparse

def extract_gnd_id(code: str) -> str:
    """Extract '4003694-7' from 'gnd:4003694-7'"""
    return code.split(':')[-1].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--label_json", type=str, required=True)  # GND-Subjects-all.json
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load full label set and build gnd_id -> index map
    with open(args.label_json, 'r', encoding='utf-8') as f:
        all_labels = json.load(f)
    
    all_gnd_ids = set(extract_gnd_id(item["Code"]) for item in all_labels)
    gnd_to_idx = {gnd: idx for idx, gnd in enumerate(sorted(all_gnd_ids))}
    #print(gnd_to_idx)
    print(f"Total labels in full set: {len(gnd_to_idx)}")

    # Save full label map
    with open(os.path.join(args.output_dir, "full_label_map.json"), "w") as f:
        json.dump(gnd_to_idx, f, ensure_ascii=False, indent=4)

    # Step 2: Process train/test data
    def process_data(json_path, split_name):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts, label_indices_list = [], []
        for item in data:
            query = item["query"]
            title = query.split("Abstract:")
            title,abstract = title[0].split("Title:")[-1].strip(),title[1].strip()
            texts.append(title + ":" + abstract)
            # texts.append(item["text"])positive_gndids
            positive_gndids=item["positive_gndids"]
            positive_gndids=[g.split(":")[-1].strip() for g in positive_gndids]
            gnds = [g for g in positive_gndids if g in gnd_to_idx]  # Filter valid

            indices = [str(gnd_to_idx[g]) for g in gnds]

            label_indices_list.append(" ".join(indices))
        return texts, label_indices_list

    train_texts, train_labels = process_data(args.train_json, "train")
    test_texts, test_labels = process_data(args.test_json, "test")

    # Save processed data
    pd.DataFrame({"text": train_texts, "labels": train_labels}).to_csv(
        os.path.join(args.output_dir, "train.csv"), index=False
    )
    pd.DataFrame({"text": test_texts, "labels": test_labels}).to_csv(
        os.path.join(args.output_dir, "test.csv"), index=False
    )

    # Save raw texts/labels for clustering (only train)
    with open(os.path.join(args.output_dir, "train_texts.txt"), "w") as f:
        for t in train_texts:
            f.write(t.replace("\n", " ") + "\n")
    with open(os.path.join(args.output_dir, "train_labels.txt"), "w") as f:
        for l in train_labels:
            f.write(l + "\n")

    print("✅ Preprocessing done.")

if __name__ == "__main__":
    main()
