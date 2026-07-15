import random
import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_ner(y_true, y_pred, plot=True, save_path=None):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("P:", p)
    print("R:", r)
    print("F1:", f1)
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    return f1, p, r


def decode_predict(input_ids, preds, labels, attention_mask, id2label):
    all_pred = []
    all_label = []
    for i in range(len(input_ids)):
        ps = []
        ls = []
        L = attention_mask[i].sum().item()
        for j in range(L):
            if labels[i][j].item() != -100:
                ps.append(id2label.get(preds[i][j].item(), 'O'))
                ls.append(id2label.get(labels[i][j].item(), 'O'))
        all_pred.append(ps)
        all_label.append(ls)
    return all_pred, all_label