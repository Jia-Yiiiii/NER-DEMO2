# 基于 BERT 的中文命名实体识别

本项目使用 BERT 模型在 MSRA 和 Weibo 两个中文数据集上进行命名实体识别（NER）实验，并对比了不同模型和标签对齐策略的效果。

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

MSRA 训练集标签分布：

| 标签 | 数量 |
|------|------|
| O | 206,412 |
| I-ORG | 9,141 |
| I-LOC | 5,313 |
| B-LOC | 3,952 |
| I-PER | 3,612 |
| B-ORG | 2,158 |
| B-PER | 1,850 |

Weibo 训练集标签分布：

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
| 1 | 数据读取 | 兼容 MSRA（0 分隔）和 Weibo（空行分隔）两种格式 |
| 2 | 标签映射 | 自动从数据中构建 label2id，MSRA 和 Weibo 分别生成 |
| 3 | 子词对齐 | BERT 分词产生子词，使用 word_ids 对齐，只保留词首子词的标签 |

对齐策略说明：

| 策略 | 说明 |
|------|------|
| ignore | 只保留词首子词的标签，其余子词忽略（设为 -100） |
| same | 词首子词用原始标签，后续子词复制标签（B- 转为 I-） |

---

## 二、实验结果

### 2.1 MSRA 数据集

#### (1) bert-base-chinese

运行命令：

```bash
python trainer.py configs/Bert_Config_exp4.json
```

测试集结果：

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| LOC | 0.9444 | 0.9130 | 0.9284 | 632 |
| ORG | 0.8500 | 0.8881 | 0.8686 | 268 |
| PER | 0.9537 | 0.9695 | 0.9615 | 361 |
| micro avg | 0.9261 | 0.9239 | 0.9250 | 1261 |
| macro avg | 0.9160 | 0.9235 | 0.9198 | 1261 |

训练曲线：

![训练损失](https://github.com/user-attachments/assets/f3e40a8a-8628-460c-ac8d-ef4bbe1bbc93)

![验证集F1](https://github.com/user-attachments/assets/830c2fd8-e166-4a00-8137-73199fae30ac)

![验证集损失](https://github.com/user-attachments/assets/2429c10a-4d5f-4d2b-b051-0e727f71f234)

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/9e6lwqi6/overview

#### (2) chinese-bert-wwm

运行命令：

```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| LOC | 0.9341 | 0.8972 | 0.9153 | 632 |
| ORG | 0.8783 | 0.8619 | 0.8701 | 268 |
| PER | 0.9420 | 0.9446 | 0.9433 | 361 |
| micro avg | 0.9245 | 0.9033 | 0.9138 | 1261 |
| macro avg | 0.9181 | 0.9012 | 0.9096 | 1261 |

训练曲线：

![训练损失](https://github.com/user-attachments/assets/bc57830c-9ccc-4c11-a28b-9e7f4b020e65)

![验证集F1](https://github.com/user-attachments/assets/b8b9dc5f-3603-4270-b096-422bc92707d1)

![验证集损失](https://github.com/user-attachments/assets/bad83de4-f5db-40e7-8c74-232a27eb42bb)

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/jzn297xe/overview

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：

```bash
python trainer.py configs/Bert_Config_exp1.json
```

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

训练曲线：

![训练损失](https://github.com/user-attachments/assets/163aa6d9-b01e-4b6b-8b5c-6704b2c12ebe)

![验证集F1](https://github.com/user-attachments/assets/275c34cc-77fc-4aec-9f25-0601120ad471)

![验证集损失](https://github.com/user-attachments/assets/2bcb5179-1105-4e71-a5bc-2718df69db4c)

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/5231tc0b/overview

#### (2) chinese-bert-wwm (ignore)

运行命令：

```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| GPE.NAM | 0.7593 | 0.8913 | 0.8200 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.5000 | 0.2632 | 0.3448 | 19 |
| LOC.NOM | 0.1667 | 0.1111 | 0.1333 | 9 |
| ORG.NAM | 0.4545 | 0.5128 | 0.4819 | 39 |
| ORG.NOM | 0.5000 | 0.4375 | 0.4667 | 16 |
| PER.NAM | 0.7453 | 0.7182 | 0.7315 | 110 |
| PER.NOM | 0.6707 | 0.6707 | 0.6707 | 167 |
| micro avg | 0.6608 | 0.6495 | 0.6551 | 408 |
| macro avg | 0.4746 | 0.4506 | 0.4623 | 408 |

训练曲线：

![训练损失](https://github.com/user-attachments/assets/e98eaece-201d-40a2-91ad-be06bc8a67c7)

![验证集F1](https://github.com/user-attachments/assets/a905bfdb-41d1-488a-ab52-df42a64e4837)

![验证集损失](https://github.com/user-attachments/assets/9dabf1c3-0eee-484a-adbe-c1bbd15de150)

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/ufj1wboz/overview

#### (3) chinese-bert-wwm (other)

运行命令：

```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| GPE.NAM | 0.7593 | 0.8913 | 0.8200 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.5000 | 0.2632 | 0.3448 | 19 |
| LOC.NOM | 0.1667 | 0.1111 | 0.1333 | 9 |
| ORG.NAM | 0.4545 | 0.5128 | 0.4819 | 39 |
| ORG.NOM | 0.5000 | 0.4375 | 0.4667 | 16 |
| PER.NAM | 0.7453 | 0.7182 | 0.7315 | 110 |
| PER.NOM | 0.6707 | 0.6707 | 0.6707 | 167 |
| micro avg | 0.6608 | 0.6495 | 0.6551 | 408 |
| macro avg | 0.4746 | 0.4506 | 0.4623 | 408 |

训练曲线：

![训练损失](https://github.com/user-attachments/assets/a89f7f4e-654f-4d79-933c-0e9280892721)

![验证集F1](https://github.com/user-attachments/assets/e7a63086-64fc-40bd-9ed5-5598531ca48a)

![验证集损失](https://github.com/user-attachments/assets/c7b72f59-2dff-4c6f-ac6e-b104bbd1597d)

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/qh1l6lak/overview

---

## 三、项目结构

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
├── checkpoints/
├── data.py
├── model.py
├── trainer.py
├── utils.py
├── label2id.json
├── requirements.txt
└── README.md
```
