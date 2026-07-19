import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = config['max_len']
        self.align_type = config.get('align_type', 'ignore')
        self.data = self.read_data(config['data_path'])
        if 'label2id' in config and config['label2id'] is not None:
            self.label2id = config['label2id']
            self.id2label = {v: k for k, v in self.label2id.items()}
        else:
            labels = set(['O'])
            for tokens, label_list in self.data:
                for label in label_list:
                    labels.add(label)
            label_list = sorted(labels)
            self.label2id = {}
            self.id2label = {}
            for i in range(len(label_list)):
                self.label2id[label_list[i]] = i
                self.id2label[i] = label_list[i]

    def read_data(self, file_path):
        data = []
        tokens = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '' or line == '0':
                    if tokens:
                        data.append((tokens, labels))
                        tokens = []
                        labels = []
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[1] if parts[1] != '0' else 'O')
        if tokens:
            data.append((tokens, labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]

        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        word_ids = tokenized.word_ids()
        L = len(word_ids)

        label_ids = []
        prev_wid = -1

        for i in range(L):
            wid = word_ids[i]

            if wid is None:
                label_ids.append(-100)
                continue

            if wid != prev_wid:
                prev_wid = wid
                if wid < len(tags):
                    label = tags[wid]
                else:
                    label = 'O'
                label_ids.append(self.label2id.get(label, -100))
                continue

            if self.align_type == 'same':
                if wid < len(tags):
                    label = tags[wid]
                    if label.startswith('B-'):
                        label = 'I-' + label[2:]
                    label_ids.append(self.label2id.get(label, -100))
                else:
                    label_ids.append(self.label2id.get('O', -100))
            else:
                label_ids.append(-100)

        input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        return input_ids, attention_mask, label_tensor

    def collate_fn(self, batch):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        for item in batch:
            all_input_ids.append(item[0])
            all_attention_masks.append(item[1])
            all_labels.append(item[2])
        return torch.stack(all_input_ids), torch.stack(all_attention_masks), torch.stack(all_labels)

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)



if __name__ == "__main__":

    config = {
        'data_path': 'data/train.txt',
        'max_len': 128,
        'align_type': 'ignore',
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    dataset = NERDataset(config, tokenizer)
    loader = dataset.get_loader(batch_size=32)
    for batch in loader:
        input_ids, attention_mask, labels = batch
        print(input_ids.shape)
        break
