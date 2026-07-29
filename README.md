# 基于 BERT 的中文命名实体识别

本项目使用 BERT 模型在 MSRA 和 Weibo 两个中文数据集上进行命名实体识别（NER）实验，并对比了不同模型和标签对齐策略的效果。

---

## 预训练模型本地下载与加载

本项目采用**本地路径加载预训练权重**。若直接调用 `from_pretrained("模型名称")`，`transformers` 库默认自动下载缓存文件，会同时拉取 PyTorch、TensorFlow 等多框架权重，存在大量冗余文件。手动下载仅保留 PyTorch 运行必需文件，磁盘占用更小，保证运行环境统一。

项目使用两组预训练模型：
- `bert-base-chinese`
- `hfl/chinese-bert-wwm`

安装下载工具：
```bash
pip install -U "huggingface_hub[cli]"

# 下载 bert-base-chinese
huggingface-cli download bert-base-chinese \
  --local-dir ./bert-base-chinese \
  --local-dir-use-symlinks False

# 下载 hfl/chinese-bert-wwm
huggingface-cli download hfl/chinese-bert-wwm \
  --local-dir ./hfl-chinese-bert-wwm \
  --local-dir-use-symlinks False
```

---

## 一、数据分析

### 1.1 数据格式

数据加载通过 `split('\n\n')` 按空行分隔句子；标签中的 `'0'` 在读取时自动转换为 `'O'`。标签映射优先从配置读取预定义的 `label2id`，若不存在则自动从数据中提取所有标签建立映射。子词对齐通过 `align_type` 参数控制，默认为 `'ignore'` 将非首子词标签设为 `-100`，若设为 `'same'` 则对后续子词使用 `I-` 标签。

两个数据集的标签体系不同：

| 数据集 | 标签格式 | 标签数量 |
|--------|---------|---------|
| MSRA | `B-LOC` 形式 | 7 种 |
| Weibo | `B-LOC.NAM` / `B-LOC.NOM` | 17 种 |

代码为两个数据集分别建立独立的 `label2id` 映射，不共用标签体系。

---

### 1.2 标签分布

**MSRA 训练集标签分布：**

| 标签 | 数量 |
|------|------|
| O | 206,412 |
| I-ORG | 9,141 |
| I-LOC | 5,313 |
| B-LOC | 3,952 |
| I-PER | 3,612 |
| B-ORG | 2,158 |
| B-PER | 1,850 |

**Weibo 训练集标签分布：**

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

---

### 1.3 数据处理流程

| 步骤 | 操作 | 说明 |
|:---:|:---:|---|
| 1 | 数据读取 | 通过 `split('\n\n')` 按空行分隔句子，`'0'` 自动转为 `'O'` |
| 2 | 标签映射 | 优先从配置读取预定义 `label2id`，否则自动从数据中提取 |
| 3 | 子词对齐 | 使用 `word_ids` 对齐，由 `align_type` 参数控制策略 |

**对齐策略说明：**

| 策略 | 说明 |
|:---:|---|
| `ignore` | 只保留词首子词的标签，其余子词忽略（设为 `-100`） |
| `same` | 词首子词用原始标签，后续子词复制标签（`B-` 转为 `I-`） |

---

## 二、实验结果

### 2.1 MSRA 数据集

#### (1) bert-base-chinese

运行命令：
```bash
python trainer.py configs/Bert_Config_exp4.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|--------|
| LOC | 0.8949 | 0.8892 | 0.8921 | 632 |
| ORG | 0.8606 | 0.8060 | 0.8324 | 268 |
| PER | 0.9344 | 0.9474 | 0.9409 | 361 |
| micro | 0.8996 | 0.8882 | 0.8939 | 1261 |
| macro | 0.8966 | 0.8809 | 0.8884 | 1261 |

训练曲线：

<img width="490" height="355" alt="image" src="https://github.com/user-attachments/assets/a1adec38-34a9-40fe-a31b-4d68d6522b15" />

<img width="1537" height="647" alt="image" src="https://github.com/user-attachments/assets/0939770f-f4ea-441d-93f4-159e66918d49" />

<img width="1522" height="688" alt="image" src="https://github.com/user-attachments/assets/d457a950-ad78-4b98-bbac-c0a196c0018d" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/vsp3un4p/overview

---

#### (2) chinese-bert-wwm

运行命令：
```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|--------|
| LOC | 0.9468 | 0.9019 | 0.9238 | 632 |
| ORG | 0.8429 | 0.8806 | 0.8613 | 268 |
| PER | 0.9346 | 0.9501 | 0.9423 | 361 |
| micro | 0.9199 | 0.9112 | 0.9155 | 1261 |
| macro | 0.9081 | 0.9109 | 0.9091 | 1261 |

训练曲线：

<img width="515" height="302" alt="image" src="https://github.com/user-attachments/assets/24e396d6-7a82-445c-9aac-130ce8256780" />

<img width="1580" height="587" alt="image" src="https://github.com/user-attachments/assets/dd9ed513-fb77-424b-a16c-6ef2b382fb15" />

<img width="1566" height="635" alt="image" src="https://github.com/user-attachments/assets/7e7a335e-b94d-46da-a632-7828d320a90d" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/wmldk5na/overview

---

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：
```bash
python trainer.py configs/Bert_Config_exp1.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|--------|
| GPE.NAM | 0.7288 | 0.9348 | 0.8190 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.3684 | 0.3684 | 0.3684 | 19 |
| LOC.NOM | 0.5000 | 0.3333 | 0.4000 | 9 |
| ORG.NAM | 0.5667 | 0.4359 | 0.4928 | 39 |
| ORG.NOM | 0.6667 | 0.3750 | 0.4800 | 16 |
| PER.NAM | 0.7094 | 0.7545 | 0.7313 | 110 |
| PER.NOM | 0.6769 | 0.7904 | 0.7293 | 167 |
| micro | 0.6690 | 0.7132 | 0.6904 | 408 |
| macro | 0.5271 | 0.4990 | 0.5026 | 408 |

训练曲线：

<img width="522" height="291" alt="image" src="https://github.com/user-attachments/assets/8d8af2ca-1207-4840-bd2d-37416ec67aa8" />

<img width="1566" height="597" alt="image" src="https://github.com/user-attachments/assets/81a9367b-4f94-4c81-8aab-020d95645fa2" />

<img width="1572" height="640" alt="image" src="https://github.com/user-attachments/assets/74e27fff-079c-4fd7-b178-62af69e9571c" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/5x5wmc18/chart

---

#### (2) chinese-bert-wwm (ignore)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|--------|
| GPE.NAM | 0.7736 | 0.8913 | 0.8283 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.3684 | 0.3684 | 0.3684 | 19 |
| LOC.NOM | 0.4000 | 0.2222 | 0.2857 | 9 |
| ORG.NAM | 0.4286 | 0.4615 | 0.4444 | 39 |
| ORG.NOM | 0.5625 | 0.5625 | 0.5625 | 16 |
| PER.NAM | 0.7250 | 0.7909 | 0.7565 | 110 |
| PER.NOM | 0.6387 | 0.7305 | 0.6816 | 167 |
| micro | 0.6413 | 0.7010 | 0.6698 | 408 |
| macro | 0.4871 | 0.5034 | 0.4909 | 408 |

训练曲线：

<img width="512" height="288" alt="image" src="https://github.com/user-attachments/assets/43f49130-2d2f-4274-bdae-a58e04503214" />

<img width="1565" height="587" alt="image" src="https://github.com/user-attachments/assets/8f0dd318-6f62-47e7-a1e1-8b6994c3df4a" />

<img width="1567" height="635" alt="image" src="https://github.com/user-attachments/assets/7d3c0f51-2d96-407f-b9d5-df2dd3f11424" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/3nxceend/overview

---

#### (3) chinese-bert-wwm (same)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|--------|
| GPE.NAM | 0.8000 | 0.8696 | 0.8333 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.4375 | 0.3684 | 0.4000 | 19 |
| LOC.NOM | 0.3333 | 0.1111 | 0.1667 | 9 |
| ORG.NAM | 0.4722 | 0.4359 | 0.4533 | 39 |
| ORG.NOM | 0.5263 | 0.6250 | 0.5714 | 16 |
| PER.NAM | 0.7642 | 0.7364 | 0.7500 | 110 |
| PER.NOM | 0.6685 | 0.7246 | 0.6954 | 167 |
| micro | 0.6740 | 0.6789 | 0.6764 | 408 |
| macro | 0.5003 | 0.4839 | 0.4838 | 408 |

训练曲线：

<img width="518" height="292" alt="image" src="https://github.com/user-attachments/assets/28f22d73-69ca-4daf-adc3-35ceec566cda" />

<img width="1567" height="593" alt="image" src="https://github.com/user-attachments/assets/37e84c90-775b-456d-a55e-ddaa4142834e" />

<img width="1553" height="637" alt="image" src="https://github.com/user-attachments/assets/0f4b20ef-0ca0-4afe-a5ce-33f578fa2e2f" />

实验日志：https://swanlab.cn/@2225/bert-ner1/runs/xjq4augx/overview

---

## 三、项目结构

```text
BERT-NER-DEMO2/
├── data/
│   ├── MSRA/
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   └── weibo/
│       ├── train.txt
│       ├── dev.txt
│       └── test.txt
├── configs/
│   ├── Bert_Config_exp1.json
│   ├── Bert_Config_exp2.json
│   ├── Bert_Config_exp3.json
│   ├── Bert_Config_exp4.json
│   ├── Bert_Config_exp5.json
│   └── label2id.json
├── data.py
├── model.py
├── trainer.py
├── utils.py
├── requirements.txt
└── README.md
```
