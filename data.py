import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NERDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = config['max_len']
        self.align_type = config.get('align_type', 'ignore')
        self.data = self.read_data(config['data_path'])
        self.label2id, self.id2label = self.get_label_map(config)

    def read_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for sentence in content.strip().split('\n\n'):
            if not sentence:
                continue
            tokens, labels = [], []
            for line in sentence.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[1] if parts[1] != '0' else 'O')
            if tokens:
                data.append((tokens, labels))
        return data

    def get_label_map(self, config):
        if 'label2id' in config and config['label2id'] is not None:
            label2id = config['label2id']
            id2label = {}
            for k, v in label2id.items():
                id2label[v] = k
            return label2id, id2label

        labels = {'O'}
        for _, tag_list in self.data:
            for tag in tag_list:
                labels.add(tag)

        labels = sorted(labels)
        label2id = {}
        id2label = {}
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label

        return label2id, id2label

    def align_labels(self, word_ids, tags):
        label_ids = []
        prev_wid = -1

        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
                continue

            if wid != prev_wid:
                prev_wid = wid
                label = tags[wid] if wid < len(tags) else 'O'
                label_ids.append(self.label2id.get(label, -100))
            else:
                if self.align_type == 'same':
                    label = tags[wid] if wid < len(tags) else 'O'
                    if label.startswith('B-'):
                        label = 'I-' + label[2:]
                    label_ids.append(self.label2id.get(label, -100))
                else:
                    label_ids.append(-100)

        return label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            return_tensors=None
        )

        word_ids = tokenized.word_ids()
        label_ids = self.align_labels(word_ids, tags)

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': label_ids,
            'length': len(tokenized['input_ids'])
        }

    def collate_fn(self, batch):
        max_len = max(item['length'] for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item['input_ids'])

            input_ids.append(
                item['input_ids'] + [0] * pad_len
            )
            attention_masks.append(
                item['attention_mask'] + [0] * pad_len
            )
            labels.append(
                item['labels'] + [-100] * pad_len
            )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
