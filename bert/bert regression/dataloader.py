
from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(
        self,
        text_field,
        label_field, 
        tokenizer,
        maxlen=256
    ):
        self._text = text_field
        self._label = label_field
        self._tokenizer = tokenizer
        self._n = len(self._text)
        self._maxlen = maxlen

    def __len__(self):
        return self._n

    def __getitem__(self, idx):

        encoding = self._tokenizer(
                            self._text[idx],
                            add_special_tokens=True,
                            max_length=self._maxlen,
                            return_token_type_ids=False,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=False,
                            return_tensors='pt',
        )
        encoding['labels'] = torch.Tensor(float(self._label[idx]))

        return encoding