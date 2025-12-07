Task: https://sites.google.com/view/llms4subjects/home

Data: https://github.com/jd-coderepos/llms4subjects/


**总损失 = 平均查询损失 + λ_reg × 多样性正则化损失**

其中：

**平均查询损失**：
对于每个查询q，损失为：
\[
L_q = L_{pos} + \lambda_{weight} \times L_{neg}
\]

**正样本损失**（平方距离损失）：
\[
L_{pos} = \frac{1}{|P_q|} \sum_{p \in P_q} (1 - \cos(q, p))^2
\]
其中 \(P_q\) 是查询q的正样本集合，\(\cos(q, p)\) 是查询q和正样本p的余弦相似度

**负样本损失**（带有边界的hinge loss）：
\[
L_{neg} = \frac{1}{|N_q|} \sum_{n \in N_q} \max(0, \alpha - (1 - \cos(q, n)))
\]
或等价地：
\[
L_{neg} = \frac{1}{|N_q|} \sum_{n \in N_q} \max(0, \alpha - d(q, n))
\]
其中 \(N_q\) 是查询q的负样本集合，\(d(q, n) = 1 - \cos(q, n)\) 是余弦距离，α是边界参数

**多样性正则化损失**：
\[
L_{reg} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|P_q|(|P_q|-1)} \sum_{i \neq j} (\cos(p_i, p_j) - 0.5)^2
\]
其中 \(p_i, p_j\) 是同一查询的不同正样本

**最终损失**：
\[
L = \frac{1}{|Q|} \sum_{q \in Q} L_q + \lambda_{reg} \times L_{reg}
\]

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
temperature=0.1        # InfoNCE温度参数
```

### 3. 数据组织形式：
- 每个查询有多个正样本（max_positives=3）
- 每个查询有多个负样本（max_negatives=8）
- 通过query_indices将文本映射回对应的查询


这个损失函数设计特别适用于多正样本、多负样本的对比学习场景，通过多样性的正则化项避免模型过度专注于特定类型的正样本，提高嵌入的泛化能力。