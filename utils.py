import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_entities(labels):
    entities = []
    i = 0
    while i < len(labels):
        if labels[i].startswith('B-'):
            entity_type = labels[i][2:]
            start = i
            i += 1
            while i < len(labels) and labels[i].startswith('I-') and labels[i][2:] == entity_type:
                i += 1
            entities.append((entity_type, start, i - 1))
        else:
            i += 1
    return set(entities)


def Ner_metrics(y_true, y_pred):
    total_tp = 0
    total_pred = 0
    total_true = 0
    for true_seq, pred_seq in zip(y_true, y_pred):
        true_entities = extract_entities(true_seq)
        pred_entities = extract_entities(pred_seq)
        tp = len(true_entities & pred_entities)
        total_tp += tp
        total_pred += len(pred_entities)
        total_true += len(true_entities)
    eps = 1e-8
    precision = total_tp / (total_pred + eps)
    recall = total_tp / (total_true + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1, total_tp, total_pred, total_true


def Ner_report(y_true, y_pred):
    p, r, f1, tp, pred_sum, true_sum = Ner_metrics(y_true, y_pred)
    return f1, p, r


def evaluate_ner(y_true, y_pred, plot=True, save_path=None):
    f1, p, r =  Ner_report(y_true, y_pred)
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
