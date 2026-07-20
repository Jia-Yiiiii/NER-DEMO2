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
| LOC    | 0.9230   | 0.9098   | 0.9163 | 632    |
| ORG    | 0.8723   | 0.8918   | 0.8819 | 268    |
| PER    | 0.9528   | 0.9501   | 0.9515 | 361    |
| micro  | 0.9204   | 0.9175   | 0.9190 | 1261   |
| macro  | 0.9160   | 0.9172   | 0.9166 | 1261   |

训练曲线：

<img width="547" height="335" alt="image" src="https://github.com/user-attachments/assets/cf2f6e1c-b7ea-449e-8ce4-7cbf5e5db72c" />
<img width="1537" height="658" alt="image" src="https://github.com/user-attachments/assets/4bb5e3ee-5387-44a4-831c-09ee9cd9a0e9" />
<img width="1578" height="647" alt="image" src="https://github.com/user-attachments/assets/1d356e79-2ccf-4740-af33-520ca03a60be" />


实验日志：https://swanlab.cn/@2225/bert-ner1/runs/g7iuacl7/overview

#### (2) chinese-bert-wwm

运行命令：

```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：

| 类型   | 精确率   | 召回率   | F1     | 样本数 |
|--------|----------|----------|--------|--------|
| LOC    | 0.9262   | 0.9130   | 0.9195 | 632    |
| ORG    | 0.8828   | 0.8993   | 0.8909 | 268    |
| PER    | 0.9501   | 0.9501   | 0.9501 | 361    |
| micro  | 0.9236   | 0.9207   | 0.9222 | 1261   |
| macro  | 0.9197   | 0.9208   | 0.9202 | 1261   |

训练曲线：


<img width="613" height="296" alt="image" src="https://github.com/user-attachments/assets/9b7e1f8c-783a-419a-83e3-27c9822c8b5d" />
<img width="1562" height="602" alt="image" src="https://github.com/user-attachments/assets/d3aeeeac-755f-46b5-b6dc-3ba35a1f07a9" />
<img width="1588" height="655" alt="image" src="https://github.com/user-attachments/assets/0157f0f5-42ba-42d4-b96e-c2ce3c1f563b" />

实验日志：[https://swanlab.cn/@2225/bert-ner1/runs/jzn297xe/overview](https://swanlab.cn/@2225/bert-ner1/runs/o79jmduy/overview)

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：

```bash
python trainer.py configs/Bert_Config_exp1.json
```

测试集结果：

| 类型     | 精确率   | 召回率   | F1     | 样本数 |
|----------|----------|----------|--------|--------|
| GPE.NAM  | 0.7593   | 0.8913   | 0.8200 | 46     |
| GPE.NOM  | 0.0000   | 0.0000   | 0.0000 | 2    |
| LOC.NAM  | 0.4286   | 0.3158   | 0.3636 | 19     |
| LOC.NOM  | 0.2500   | 0.2222   | 0.2353 | 9      |
| ORG.NAM  | 0.4516   | 0.3590   | 0.4000 | 39     |
| ORG.NOM  | 0.4286   | 0.1875   | 0.2609 | 16     |
| PER.NAM  | 0.7179   | 0.7636   | 0.7401 | 110    |
| PER.NOM  | 0.6927   | 0.7425   | 0.7168 | 167    |
| micro    | 0.6683   | 0.6716   | 0.6699 | 408    |
| macro    | 0.4661   | 0.4352   | 0.4421 | 408    |

训练曲线：


<img width="567" height="361" alt="image" src="https://github.com/user-attachments/assets/46a95a90-ccfd-48a2-a48f-c8606c220a45" />
<img width="1532" height="703" alt="image" src="https://github.com/user-attachments/assets/bd11e2b2-6bec-4d7c-b78c-2727b45dabd4" />
<img width="1572" height="642" alt="image" src="https://github.com/user-attachments/assets/f93a5dd8-f58a-4eda-8bef-b1605ee15e1e" />


实验日志：[https://swanlab.cn/@2225/bert-ner1/runs/5231tc0b/overview](https://swanlab.cn/@2225/bert-ner1/runs/roxd8y4m/overview)

#### (2) chinese-bert-wwm (ignore)

运行命令：

```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：

| 类型     | 精确率   | 召回率   | F1     | 样本数 |
|----------|----------|----------|--------|--------|
| GPE.NAM  | 0.7736   | 0.8913   | 0.8283 | 46     |
| GPE.NOM  | 0.0000   | 0.0000   | 0.0000 | 2      |
| LOC.NAM  | 0.4375   | 0.3684   | 0.4000 | 19     |
| LOC.NOM  | 0.2500   | 0.1111   | 0.1538 | 9      |
| ORG.NAM  | 0.5758   | 0.4872   | 0.5278 | 39     |
| ORG.NOM  | 0.5333   | 0.5000   | 0.5161 | 16     |
| PER.NAM  | 0.6810   | 0.7182   | 0.6991 | 110    |
| PER.NOM  | 0.6919   | 0.7126   | 0.7021 | 167    |
| micro    | 0.6699   | 0.6716   | 0.6707 | 408    |
| macro    | 0.4929   | 0.4736   | 0.4784 | 408    |

训练曲线：

<img width="541" height="328" alt="image" src="https://github.com/user-attachments/assets/59204772-18cb-4367-8261-08248c0731b0" />
<img width="1533" height="666" alt="image" src="https://github.com/user-attachments/assets/a83f04c4-eb17-473d-8249-40b5cb390c22" />
<img width="1522" height="697" alt="image" src="https://github.com/user-attachments/assets/1b45ce08-234b-4c88-bb12-9b8d845aa653" />


实验日志：https://swanlab.cn/@2225/bert-ner1/runs/hdqsppbv/chart
#### (3) chinese-bert-wwm (other)

运行命令：

```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：

| 类型     | 精确率   | 召回率   | F1     | 样本数 |
|----------|----------|----------|--------|--------|
| GPE.NAM  | 0.7736   | 0.8913   | 0.8283 | 46     |
| GPE.NOM  | 0.0000   | 0.0000   | 0.0000 | 2      |
| LOC.NAM  | 0.4375   | 0.3684   | 0.4000 | 19     |
| LOC.NOM  | 0.2500   | 0.1111   | 0.1538 | 9      |
| ORG.NAM  | 0.5758   | 0.4872   | 0.5278 | 39     |
| ORG.NOM  | 0.5333   | 0.5000   | 0.5161 | 16     |
| PER.NAM  | 0.6810   | 0.7182   | 0.6991 | 110    |
| PER.NOM  | 0.6919   | 0.7126   | 0.7021 | 167    |
| micro    | 0.6699   | 0.6716   | 0.6707 | 408    |
| macro    | 0.4929   | 0.4736   | 0.4784 | 408    |



训练曲线：



<img width="557" height="328" alt="image" src="https://github.com/user-attachments/assets/901a14f2-db6c-46f5-b0a2-f5e3056fc45d" />

<img width="1542" height="666" alt="image" src="https://github.com/user-attachments/assets/6f8499e3-44d9-4339-b4c0-fdd72b872704" />

<img width="1538" height="690" alt="image" src="https://github.com/user-attachments/assets/bc98ab94-3bd5-437a-a414-31040b4835de" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/ivg1r5lz/overview

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
