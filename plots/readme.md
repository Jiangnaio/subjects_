```python
import json
from transformers import AutoTokenizer

train_path="datasets/qwen3_embedding_train.json"
with open(train_path, 'r') as f:
    samples = json.load(f)
#画图统计samples的query的token长度分布
import matplotlib.pyplot as plt
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
query_token_lengths = [len(tokenizer(sample['query']).input_ids) for sample in samples]
plt.hist(query_token_lengths, bins=50, edgecolor='black')
plt.xlabel('Query Token Length')
plt.ylabel('Number of Samples')
plt.title('Query Token Length Distribution')
plt.show()

```
