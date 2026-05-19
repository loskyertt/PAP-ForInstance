# 第22课：词嵌入与语义映射

## 1. 本节学习目标

- 理解词嵌入（Word Embedding）在零样本学习中的作用
- 掌握 GloVe 的加载和使用方式
- 理解 `dataloaders/get_embedding.py` 脚本的工作流程
- 能够为新的数据集生成词嵌入

---

## 2. 零样本学习的核心挑战

### 问题

```
Seen Classes (有标注):
  wall, floor, chair, table, desk, bed, bookshelf, sofa, sink, bathtub

Unseen Classes (无标注):
  toilet, curtain, counter, door, window, ...

目标: 模型要在无标注的情况下识别 unseen classes
```

### 语义桥接

**唯一信息**：类别的名称！

```
"curtain" 这个词本身就包含了语义信息:
  - 和 "window" 相关
  - 和 "wall" 可能有关系
  - 和 "bathtub" 无关

→ 用词嵌入将"名称"映射为"语义向量"
→ 在语义空间中桥接 seen 和 unseen
```

---

## 3. GloVe 词嵌入

### 什么是 GloVe

GloVe (Global Vectors for Word Representation) 是无监督学习的词向量模型：

```
"chair"  →  [0.12, -0.34, 0.56, ..., 0.78]  (300维向量)
"table"  →  [0.10, -0.30, 0.52, ..., 0.81]
"toilet" →  [-0.45, 0.23, -0.11, ..., -0.32]

语义关系:
  cos("chair", "table") ≈ 0.65   (相关! 都是家具)
  cos("chair", "toilet") ≈ 0.15  (不太相关)
```

### GloVe 的特性

| 属性 | 说明 |
|---|---|
| 维度 | 300 (glove-wiki-gigaword-300) |
| 训练语料 | Wikipedia + Gigaword (60亿词) |
| 语义保留 | 同类相近，不同类远离 |
| 类比性 | king - man + woman ≈ queen |

---

## 4. get_embedding.py 脚本解析

### 完整流程

```python
# dataloaders/get_embedding.py
import gensim.downloader as api
import numpy as np
import pickle

def get_embeddings(dataset_name, embedding_type='word2vec'):
    """
    为数据集的所有类生成词嵌入
    """
    # Step 1: 加载 GloVe 模型
    print("Loading GloVe model...")
    glove = api.load(f"glove-wiki-gigaword-{embedding_dim}")
    # 首次运行约需下载 1GB
    
    # Step 2: 获取类名列表
    if dataset_name == 'S3DIS':
        classes = S3DIS_CLASSES
    elif dataset_name == 'ScanNet':
        classes = SCANNET_CLASSES
    
    # Step 3: 为每个类生成词嵌入
    embeddings = {}
    for cls_name in classes:
        # 处理多词短语
        # e.g. "shower curtain" → mean("shower", "curtain")
        words = cls_name.lower().split()
        word_vecs = []
        for word in words:
            if word in glove:
                word_vecs.append(glove[word])
        
        if word_vecs:
            cls_embedding = np.mean(word_vecs, axis=0)
        else:
            cls_embedding = np.random.randn(embedding_dim) * 0.01
        
        embeddings[cls_name] = cls_embedding
    
    # Step 4: 保存
    save_path = f'datasets/{dataset_name}/word_embeddings.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved {len(embeddings)} embeddings to {save_path}")
    return embeddings

if __name__ == '__main__':
    get_embeddings('S3DIS')
```

### 运行方式

```bash
python dataloaders/get_embedding.py --dataset S3DIS
python dataloaders/get_embedding.py --dataset ScanNet
```

---

## 5. 多词类名的处理

### 挑战

```
"shower curtain" → 如何映射为一个向量？
"other furniture" → 在 GloVe 中可能不存在
```

### 处理策略

```python
def get_multiword_embedding(phrase, glove_model):
    words = phrase.lower().split()
    
    vectors = []
    for word in words:
        if word in glove_model:
            vectors.append(glove_model[word])
    
    if vectors:
        # 策略1: 平均 (本项目使用)
        return np.mean(vectors, axis=0)
        
        # 策略2: 加权平均 (按 IDF 权重)
        # weights = [idf[w] for w in words]
        # return np.average(vectors, weights=weights, axis=0)
        
        # 策略3: 只用第一个词
        # return vectors[0]
    else:
        # 回退: 随机初始化
        return np.random.randn(300) * 0.01
```

---

## 6. 嵌入空间可视化

### t-SNE 降维可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

# 加载嵌入
with open('datasets/S3DIS/word_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# 提取向量和标签
labels = list(embeddings.keys())
vectors = np.array([embeddings[l] for l in labels])

# t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, label in enumerate(labels):
    plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]))
plt.title('S3DIS Class Word Embeddings (t-SNE)')
plt.show()
```

### 期望结果

```
语义空间:
  家具组 (靠近):    chair, table, sofa, bookcase
  建筑组 (靠近):    wall, floor, ceiling, beam, column
  开口组 (靠近):    door, window
  家具与建筑 (远离)
```

---

## 7. 词嵌入在模型中的使用

### 在 GMMN 生成器中（下节课详述）

```python
# 在 ProtoNetAlignFZ 的 forward 中
def forward(self, support_x, support_y, query_x, query_y, class_names):
    ...
    # 获取词嵌入
    word_embeddings = self._get_embeddings(class_names)  # (C, 300)
    
    # 生成伪视觉原型 (for unseen classes)
    noise = torch.randn(C, noise_dim).cuda()
    fake_prototypes = self.generator(word_embeddings, noise)  # (C, D)
    
    # 合并真实原型和伪原型
    all_prototypes = torch.cat([real_prototypes, fake_prototypes], dim=1)
    ...
```

---

## 8. 本节总结

| 概念 | 要点 |
|---|---|
| GloVe | 300 维词向量，Wikipedia 训练 |
| 语义桥接 | 通过词向量将视觉和语言连接 |
| 多词处理 | 分词后取平均 |
| 保存格式 | pickle: `{class_name: 300-dim vector}` |
| 使用场景 | 零样本学习：从词向量生成视觉原型 |

---

## 9. 课后练习

1. 运行 `get_embedding.py`，为 S3DIS 和 ScanNet 生成词嵌入
2. 用 t-SNE 可视化两个数据集的词嵌入，分析类别分布
3. 比较不同词向量模型（GloVe, FastText, BERT）在语义相似度任务上的表现
4. 设计一个度量：给定两个类名，用词嵌入预测它们在视觉上是否容易混淆
