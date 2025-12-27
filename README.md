2025.12.27 使用info_nce该经版的损失函数，训练Snowflake/snowflake-arctic-embed-l-v2.0，采用bf16精度训练，训练过程中发现损失函数值不太稳定，训练到约0.55 epoch触发早停（即只训练55%的数据），dev测试中，recal@50从0.6697提升至0.7604。评测test集合，加上简单的关键词匹配重排序算法，test.recall@5=0.3893, test.recall@50=0.6685. 参考https://arxiv.org/pdf/2407.18887文档，优化训练数据采样与组织方式，或许能够沟达到更优的效果。

2025.12.26 加入熵正则化，虽然能够使训练的损失函数更平滑，但训练效果更差（训练0.6B参数）；训练4B参数仍然出现效果不升反降。参考Qwen3-Embedding使用的损失函数，重写损失函数（info_nce多主题改进版），对qwen3-embedding-0.6b先前训练的epoch-1的基础上，进行全参数微调，训练1500 steps(batch_size=16) 效果明显提升, 在dev随机1000样本数据集上recall@50从0.6944提升到0.7722(zero-shot为0.6434), 训练2000 steps 有所下降为0.7529。 

2025.12.25 训练4B参数时，训练后测试不升反降，分析原因在于当前损失函数鼓励 query 与正样本更近、负样本更远，但未约束输出分布的“锐度”。每个训练样本硬性设置10个随机负主题，2个硬负主题。正主题最大限制为5。同时考虑使用lora微调设置更小的r=16。使用batch_size=12训练1250 steps，效果有约2%的提升，训练1750 steps时，效果反而下降。在训练0.6B参数模型时，Lora r=32训练的时间比全参数微调更长。考虑在损失函数中加入熵正则化，鼓励模型输出更均匀、更具探索性的相似度分布，防“坍塌”（collapse）。目前训练最好的结果是训练训练了8500 steps 的qwen3-emb-0.6b(未使用熵正则化，epoch=1, seed=42,训练数据集约前55%， 完整训练2 epoch后的效果稍差0.3%)。分析训练损失函数曲线图，训练存在过早收敛，后续训练对结果影响不大。通过加入熵正则化，能够抑制训练过早收敛。recall@50可以通过微调Embedding模型来提升，目前最好为60%. recal@5最好为30%（全参微调0.6B参数的结果，4B参数的微调待优化）。 在dev数据集中随机选取1000个样本分析，发现即使标题中出现过的主题词，未必就是相关主题。

2025.12.21 分析发现，与title+abstract相关的主题很可能在title和abstract中出现过，故利用此特性进行重排序。可以明显提升结果。（见qwen3-0.6b-constractive-rank.xlsx）; 通过加入难样本微调（检索结果相似但不相关的主题），并不能提升检索效果，相反，效果下降。采用dev数据集对各个方法微调的测试结果见目录dev_eval.

2025.12.7 由于title+abstract有大量长文本，且每个文本可能存在多个相关主题，考虑使用LLM提取长文本与主题相关的原因分析。一个长文本+一个正相关主题-->总结性句子。 利用LLM生成的句子与主题对Embedding模型进行微调。正分析如何利用生成的句子来提高微调效果。
考虑加入生成任务来提高微调效果，参考《UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations》的思路，效果不佳。正分析原因。

Task: https://sites.google.com/view/llms4subjects/home

Data: https://github.com/jd-coderepos/llms4subjects/


**总损失 = 平均查询损失 + λ_reg × 多样性正则化损失**

其中：

**平均查询损失**：
对于每个查询q，损失为：
$
L_q = L_{pos} + \lambda_{weight} \times L_{neg}
$

**正样本损失**（平方距离损失）：
$
L_{pos} = \frac{1}{|P_q|} \sum_{p \in P_q} (1 - \cos(q, p))^2
$
其中 $P_q$ 是查询q的正样本集合，$\cos(q, p)$ 是查询q和正样本p的余弦相似度

**负样本损失**（带有边界的hinge loss）：
$
L_{neg} = \frac{1}{|N_q|} \sum_{n \in N_q} \max(0, \alpha - (1 - \cos(q, n)))
$
或等价地：
$
L_{neg} = \frac{1}{|N_q|} \sum_{n \in N_q} \max(0, \alpha - d(q, n))
$
其中 $N_q$ 是查询q的负样本集合，$d(q, n) = 1 - \cos(q, n)$ 是余弦距离，α是边界参数

**多样性正则化损失**：
$
L_{reg} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|P_q|(|P_q|-1)} \sum_{i \neq j} (\cos(p_i, p_j) - 0.5)^2
$
其中 $p_i, p_j$ 是同一查询的不同正样本

**最终损失**：
$
L = \frac{1}{|Q|} \sum_{q \in Q} L_q + \lambda_{reg} \times L_{reg}
$

## 损失函数特点分析

### 1. 改进对比损失的创新点：

- **平方距离损失**：对正样本使用平方距离，强调对正样本的紧密性
- **动态边界**：通过α参数控制负样本的分离边界
- **多样性正则化**：防止正样本聚集过密，鼓励多样化的表示
- **自适应调整**：α参数随训练衰减（从α_max到α_min）

### 2. 训练策略相关参数：

```
alpha_min=0.3          # 最小边界
alpha_max=0.6          # 初始边界
alpha_decay=0.9995     # 衰减率
lambda_weight=0.8      # 负样本权重
lambda_reg=0.1         # 多样性正则化权重
```

### 3. 数据组织形式：

- 每个查询有多个正样本（max_positives=3）
- 每个查询有多个负样本（max_negatives=8）
- 通过query_indices将文本映射回对应的查询


这个损失函数设计特别适用于多正样本、多负样本的对比学习场景，通过多样性的正则化项避免模型过度专注于特定类型的正样本，提高嵌入的泛化能力。
