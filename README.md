# 基于 BERT 的中文命名实体识别

本项目使用 BERT 模型在 **MSRA** 和 **Weibo** 两个中文数据集上进行命名实体识别（NER）实验，并对比了不同模型和标签对齐策略的效果。

---

## 一、数据分析

### 1.1 数据格式

MSRA 用 `0` 分隔句子，Weibo 用空行分隔。代码中通过判断 `line == '' or line == '0'` 同时兼容两种格式。

两个数据集的标签体系不同：

| 数据集 | 标签格式 | 标签数量 |
|--------|---------|---------|
| MSRA | `B-LOC` 形式 | 7 种 |
| Weibo | `B-LOC.NAM` / `B-LOC.NOM` | 17 种 |

代码为两个数据集分别建立独立的 `label2id` 映射，不共用标签体系。

### 1.2 标签分布

**MSRA 训练集标签分布**

| 标签 | 数量 |
|------|------|
| O | 206,412 |
| I-ORG | 9,141 |
| I-LOC | 5,313 |
| B-LOC | 3,952 |
| I-PER | 3,612 |
| B-ORG | 2,158 |
| B-PER | 1,850 |

**Weibo 训练集标签分布**

| 标签 | 数量 |
|------|------|
| O | 68,777 |
| I-PER.NOM | 1,043 |
| I-PER.NAM | 1,041 |
| B-PER.NOM | 766 |
| B-PER.NAM | 574 |
| I-ORG.NAM | 477 |
| I-GPE.NAM | 241 |
| B-GPE.NAM | 205 |
| B-ORG.NAM | 183 |
| I-LOC.NAM | 129 |
| I-LOC.NOM | 66 |
| I-ORG.NOM | 61 |
| B-LOC.NAM | 56 |
| B-LOC.NOM | 51 |
| B-ORG.NOM | 42 |
| B-GPE.NOM | 8 |
| I-GPE.NOM | 8 |

### 1.3 数据处理流程

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 数据读取 | 兼容 MSRA（`0` 分隔）和 Weibo（空行分隔）两种格式 |
| 2 | 标签映射 | 自动从数据中构建 `label2id`，MSRA 和 Weibo 分别生成 |
| 3 | 子词对齐 | BERT 分词产生子词，使用 `word_ids` 对齐，只保留词首子词的标签 |

**对齐策略说明**

| 策略 | 说明 |
|------|------|
| `ignore` | 只保留词首子词的标签，其余子词忽略（设为 -100） |
| `same` | 词首子词用原始标签，后续子词复制标签（B- 转为 I-） |

---

## 二、实验结果

### 整体结果对比

| 模型 | 数据集 | 对齐策略 | 最佳验证 F1 | 测试集 F1 |
|------|--------|----------|-------------|-----------|
| bert-base-chinese | MSRA | ignore | 0.9263 | 0.9192 |
| chinese-bert-wwm | MSRA | ignore | 0.9403 | 0.9095 |
| bert-base-chinese | Weibo | ignore | 0.7117 | 0.6682 |
| chinese-bert-wwm | Weibo | ignore | 0.7075 | 0.6460 |
| chinese-bert-wwm | Weibo | other | 0.7075 | 0.6460 |

### 2.1 MSRA数据集
1.bert-base-chinese

| 实体类型 | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| LOC | 94.52% | 93.93% | 94.23% | 643 |
| ORG | 80.54% | 92.26% | 86.00% | 323 |
| PER | 96.14% | 97.39% | 96.76% | 307 |
| micro avg | 90.98% | 94.34% | 92.63% | 1,273 |
| macro avg | 90.40% | 94.53% | 92.33% | 1,273 |
| weighted avg | 91.37% | 94.34% | 92.75% | 1,273 |

2. chinese-bert-wwm

**测试集详细结果**

| 实体类型 | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| LOC | 95.72% | 93.93% | 94.82% | 643 |
| ORG | 87.24% | 91.02% | 89.09% | 323 |
| PER | 97.41% | 98.05% | 97.73% | 307 |
| micro avg | 93.89% | 94.19% | 94.04% | 1,273 |
| macro avg | 93.46% | 94.33% | 93.88% | 1,273 |
| weighted avg | 93.98% | 94.19% | 94.07% | 1,273 |

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：
python trainer.py configs/Bert_Config_exp1.json

测试集结果：

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| GPE.NAM | 0.7593 | 0.8913 | 0.8200 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 0 |
| LOC.NAM | 0.4545 | 0.2632 | 0.3333 | 19 |
| LOC.NOM | 0.2500 | 0.3333 | 0.2857 | 9 |
| ORG.NAM | 0.5405 | 0.5128 | 0.5263 | 39 |
| ORG.NOM | 0.5385 | 0.4375 | 0.4828 | 16 |
| PER.NAM | 0.7281 | 0.7545 | 0.7411 | 110 |
| PER.NOM | 0.6828 | 0.7605 | 0.7195 | 167 |
| micro avg | 0.6698 | 0.7010 | 0.6850 | 408 |
| macro avg | 0.4942 | 0.4941 | 0.4942 | 408 |

**训练曲线**

| | | |
|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/163aa6d9-b01e-4b6b-8b5c-6704b2c12ebe" width="95%" /> | <img src="https://github.com/user-attachments/assets/275c34cc-77fc-4aec-9f25-0601120ad471" width="95%" /> | <img src="https://github.com/user-attachments/assets/2bcb5179-1105-4e71-a5bc-2718df69db4c" width="95%" /> |
实验日志：
https://swanlab.cn/@2225/bert-ner1/runs/5231tc0b/overview

### 2.4 Weibo + chinese-bert-wwm (ignore)

**测试集详细结果**

| 实体类型 | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| GPE.NAM | 68.75% | 84.62% | 75.86% | 26 |
| GPE.NOM | 100.00% | 100.00% | 100.00% | 1 |
| LOC.NAM | 50.00% | 83.33% | 62.50% | 6 |
| LOC.NOM | 40.00% | 33.33% | 36.36% | 6 |
| ORG.NAM | 37.50% | 44.68% | 40.78% | 47 |
| ORG.NOM | 44.44% | 80.00% | 57.14% | 5 |
| PER.NAM | 73.86% | 73.03% | 73.45% | 89 |
| PER.NOM | 77.25% | 78.37% | 77.80% | 208 |
| micro avg | 68.69% | 72.94% | 70.75% | 388 |
| macro avg | 61.48% | 72.17% | 65.49% | 388 |
| weighted avg | 69.73% | 72.94% | 71.10% | 388 |

### 2.5 Weibo + chinese-bert-wwm (other)

**测试集详细结果**

| 实体类型 | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| GPE.NAM | 68.75% | 84.62% | 75.86% | 26 |
| GPE.NOM | 100.00% | 100.00% | 100.00% | 1 |
| LOC.NAM | 50.00% | 83.33% | 62.50% | 6 |
| LOC.NOM | 40.00% | 33.33% | 36.36% | 6 |
| ORG.NAM | 37.50% | 44.68% | 40.78% | 47 |
| ORG.NOM | 44.44% | 80.00% | 57.14% | 5 |
| PER.NAM | 73.86% | 73.03% | 73.45% | 89 |
| PER.NOM | 77.25% | 78.37% | 77.80% | 208 |
| micro avg | 68.69% | 72.94% | 70.75% | 388 |
| macro avg | 61.48% | 72.17% | 65.49% | 388 |
| weighted avg | 69.73% | 72.94% | 71.10% | 388 |

---

## 三、结果分析

| 观察点 | 结论 |
|--------|------|
| MSRA vs Weibo | MSRA 上测试 F1 达 0.94，Weibo 上仅 0.66-0.70，社交媒体文本 NER 难度显著更高 |
| 模型对比 | `chinese-bert-wwm` 在 MSRA 上优于 `bert-base-chinese`（F1: 0.9404 vs 0.9263） |
| 对齐策略 | `ignore` 和 `other` 在 Weibo 上结果几乎相同，差异不大 |

---

## 四、项目结构

```text
BERT-NER-DEMO2/
├── DATA/
│   ├── MSRA/
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   └── weibo/
│       ├── train.txt
│       ├── dev.txt
│       └── test.txt
├── configs/
│   └── Bert_Config_exp1.json
├── checkpoints/          # 训练保存的模型权重
├── data.py               # 数据加载与预处理
├── model.py              # BERT 模型定义
├── trainer.py            # 训练主逻辑
├── utils.py              # 工具函数（评估、解码等）
├── label2id.json         # 标签映射文件
├── requirements.txt
└── README.md
