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

| 类型   | 精确率   | 召回率   | F1     | 样本数 |
|--------|----------|----------|--------|--------|
| LOC    | 0.9231   | 0.8924   | 0.9075 | 632    |
| ORG    | 0.8664   | 0.8470   | 0.8566 | 268    |
| PER    | 0.9368   | 0.9446   | 0.9407 | 361    |
| micro  | 0.9151   | 0.8977   | 0.9063 | 1261   |
| macro  | 0.9088   | 0.8947   | 0.9016 | 1261   |

训练曲线：

<img width="515" height="336" alt="image" src="https://github.com/user-attachments/assets/3b12616d-9d59-4d00-bd1a-a006050b3332" />
<img width="1537" height="662" alt="image" src="https://github.com/user-attachments/assets/7ddc587c-be4f-42b5-9262-801011af1b4d" />
<img width="1518" height="695" alt="image" src="https://github.com/user-attachments/assets/6eef4a58-1e2f-414f-bb77-cd9b1267b5dd" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/okdbik55/overview

#### (2) chinese-bert-wwm

运行命令：

```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：

| 类型   | 精确率   | 召回率   | F1     | 样本数 |
|--------|----------|----------|--------|--------|
| LOC    | 0.9220   | 0.9161   | 0.9190 | 632    |
| ORG    | 0.8712   | 0.8582   | 0.8647 | 268    |
| PER    | 0.9552   | 0.9446   | 0.9499 | 361    |
| micro  | 0.9207   | 0.9120   | 0.9163 | 1261   |
| macro  | 0.9161   | 0.9063   | 0.9112 | 1261   |

训练曲线：
<img width="547" height="395" alt="image" src="https://github.com/user-attachments/assets/d3fbc11d-99d4-4f9b-97ab-e28cad037917" />
<img width="1513" height="697" alt="image" src="https://github.com/user-attachments/assets/1be4bb36-8f94-4e1b-bdb5-edff061cb8e3" />
<img width="1558" height="642" alt="image" src="https://github.com/user-attachments/assets/b6fa89af-2c4a-4902-b02c-099a76e96d93" />



实验日志：https://swanlab.cn/@2225/bert-ner1/runs/ycbn1m67/overview

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：

```bash
python trainer.py configs/Bert_Config_exp1.json
```

测试集结果：

| 类型     | 精确率   | 召回率   | F1     | 样本数 |
|----------|----------|----------|--------|--------|
| GPE.NAM  | 0.7551   | 0.8043   | 0.7789 | 46     |
| GPE.NOM  | 0.0000   | 0.0000   | 0.0000 | 2      |
| LOC.NAM  | 0.3571   | 0.2632   | 0.3030 | 19     |
| LOC.NOM  | 0.2222   | 0.2222   | 0.2222 | 9      |
| ORG.NAM  | 0.6250   | 0.3846   | 0.4762 | 39     |
| ORG.NOM  | 0.7778   | 0.4375   | 0.5600 | 16     |
| PER.NAM  | 0.7182   | 0.7182   | 0.7182 | 110    |
| PER.NOM  | 0.7030   | 0.6946   | 0.6988 | 167    |
| micro    | 0.6868   | 0.6397   | 0.6624 | 408    |
| macro    | 0.5198   | 0.4406   | 0.4697 | 408    |

训练曲线：

<img width="540" height="300" alt="image" src="https://github.com/user-attachments/assets/4e3b22b4-52f9-4b0b-b624-50c51437fc38" />
<img width="1590" height="593" alt="image" src="https://github.com/user-attachments/assets/7c6a9581-104f-4aa2-9203-fe0bf02edf78" />
<img width="1577" height="640" alt="image" src="https://github.com/user-attachments/assets/4223fde4-67b9-44ee-bd46-73fa11c71e68" />


实验日志：https://swanlab.cn/@2225/bert-ner1/runs/82d9xldj/overview

#### (2) chinese-bert-wwm (ignore)

运行命令：

```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：

| 类型     | 精确率   | 召回率   | F1     | 样本数 |
|----------|----------|----------|--------|--------|
| GPE.NAM  | 0.7593   | 0.8913   | 0.8200 | 46     |
| GPE.NOM  | 0.0000   | 0.0000   | 0.0000 | 2      |
| LOC.NAM  | 0.4000   | 0.4211   | 0.4103 | 19     |
| LOC.NOM  | 0.2143   | 0.3333   | 0.2609 | 9      |
| ORG.NAM  | 0.5000   | 0.4359   | 0.4658 | 39     |
| ORG.NOM  | 0.7500   | 0.3750   | 0.5000 | 16     |
| PER.NAM  | 0.7522   | 0.7727   | 0.7623 | 110    |
| PER.NOM  | 0.6736   | 0.7784   | 0.7222 | 167    |
| micro    | 0.6651   | 0.7108   | 0.6872 | 408    |
| macro    | 0.5062   | 0.5010   | 0.4927 | 408    |

训练曲线：

<img width="535" height="302" alt="image" src="https://github.com/user-attachments/assets/be1fb583-4d8d-4857-a5c0-68611fe937ac" />
<img width="1575" height="598" alt="image" src="https://github.com/user-attachments/assets/550774ea-2df6-4f47-9e0b-1ed77310967a" />
<img width="1570" height="638" alt="image" src="https://github.com/user-attachments/assets/8f86b1fa-a0a9-4fa2-a749-cb65c91fb86c" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/pn8qxcj5/overview
#### (3) chinese-bert-wwm (same)

运行命令：

```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：

| 类型     | 精确率   | 召回率   | F1     | 样本数 |
|----------|----------|----------|--------|--------|
| GPE.NAM  | 0.7736   | 0.8913   | 0.8283 | 46     |
| GPE.NOM  | 0.0000   | 0.0000   | 0.0000 | 2      |
| LOC.NAM  | 0.3333   | 0.4211   | 0.3721 | 19     |
| LOC.NOM  | 0.4545   | 0.5556   | 0.5000 | 9      |
| ORG.NAM  | 0.4857   | 0.4359   | 0.4595 | 39     |
| ORG.NOM  | 0.5625   | 0.5625   | 0.5625 | 16     |
| PER.NAM  | 0.7264   | 0.7000   | 0.7130 | 110    |
| PER.NOM  | 0.6761   | 0.7126   | 0.6939 | 167    |
| micro    | 0.6556   | 0.6765   | 0.6659 | 408    |
| macro    | 0.5015   | 0.5349   | 0.5161 | 408    |



训练曲线：

<img width="532" height="293" alt="image" src="https://github.com/user-attachments/assets/4e5870a4-292b-4d2b-80ce-79a4fdfc2ddd" />
<img width="1557" height="592" alt="image" src="https://github.com/user-attachments/assets/54090654-88a9-4cd1-825f-37a259c9754c" />
<img width="1566" height="637" alt="image" src="https://github.com/user-attachments/assets/4d820f36-cfe4-4772-a782-440697ccffd4" />



实验日志：https://swanlab.cn/@2225/bert-ner1/runs/6x4ryoou/overview

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
│   └── Bert_Config_exp2.json
│   └── Bert_Config_exp3.json
│   └── Bert_Config_exp4.json
│   └── Bert_Config_exp5.json
├── data.py
├── model.py
├── trainer.py
├── utils.py
├── label2id.json
├── requirements.txt
└── README.md
```
