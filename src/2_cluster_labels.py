# 2_cluster_labels.py
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import json
import argparse

def get_sparse_feature(text_file, label_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    vectorizer = TfidfVectorizer(max_features=200000, dtype=np.float32)#TF-IDF将文本转换为稀疏特征向量
    X = vectorizer.fit_transform(texts)
    X = normalize(X)
    with open(label_file, 'r') as f:
        labels = [line.strip().split() for line in f]
    return X, labels

def build_balanced_kmeans(X, labels, max_leaf=125, eps=1e-4):
    # X是tfitd提取的向量，Y是多标签二维矩阵。
    mlb = MultiLabelBinarizer(sparse_output=True)
    Y = mlb.fit_transform(labels)  # [N, L_train]
    print(f"Clustering {Y.shape[1]} training labels...")

    label_features = normalize(csr_matrix(Y.T) @ csc_matrix(X))  # [L_train, D]

    q = [(np.arange(label_features.shape[0]), label_features)]
    final_groups = []

    while q:
        next_q = []
        for node_labels, node_feats in q:
            if len(node_labels) <= max_leaf:
                final_groups.append(node_labels)
            else:
                n = len(node_labels)
                c1, c2 = np.random.choice(n, 2, replace=False)
                centers = node_feats[[c1, c2]].toarray()
                old_dis = -10000.0
                new_dis = -1.0
                while new_dis - old_dis >= eps:
                    dis = node_feats @ centers.T
                    partition = np.argsort(dis[:, 1] - dis[:, 0])
                    left_idx = partition[:n//2]
                    right_idx = partition[n//2:]
                    old_dis = new_dis
                    new_dis = (dis[left_idx, 0].sum() + dis[right_idx, 1].sum()) / n
                    centers = normalize(np.array([
                        np.asarray(node_feats[left_idx].sum(axis=0)).flatten(),
                        np.asarray(node_feats[right_idx].sum(axis=0)).flatten()
                    ]))
                next_q.append((node_labels[left_idx], node_feats[left_idx]))
                next_q.append((node_labels[right_idx], node_feats[right_idx]))
        q = next_q

    # Map back to original label indices (in training set)
    train_label_indices = mlb.classes_  # [L_train]
    cluster_groups = []
    for group in final_groups:
        original_indices = train_label_indices[group]
        cluster_groups.append(original_indices.astype(int))
    return cluster_groups

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--max_leaf", type=int, default=125)
    args = parser.parse_args()

    text_file = os.path.join(args.data_dir, "train_texts.txt")
    label_file = os.path.join(args.data_dir, "train_labels.txt")
    X, labels = get_sparse_feature(text_file, label_file)

    train_clusters = build_balanced_kmeans(X, labels, max_leaf=args.max_leaf)

    # Load full label map
    with open(os.path.join(args.data_dir, "full_label_map.json")) as f:
        full_gnd_to_idx = json.load(f)
    idx_to_gnd = {v: k for k, v in full_gnd_to_idx.items()}
    n_full_labels = len(full_gnd_to_idx)

    # Build cluster assignment for ALL labels (unseen = -1)
    cluster_assign = np.full(n_full_labels, -1, dtype=int)
    for cid, group in enumerate(train_clusters):
        for idx in group:
            cluster_assign[idx] = cid

    # Save
    np.save(os.path.join(args.data_dir, "cluster_assign.npy"), cluster_assign)
    print(f"Total clusters: {len(train_clusters)}")
    print("✅ Clustering done.")

if __name__ == "__main__":
    main()
