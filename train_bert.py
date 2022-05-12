import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import CONFIG
from model.bert4rec import Albert4Rec


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['1M'], help='데이터셋', type=str)
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)
    parser.add_argument('-cpu', '--cpu', action='store_true', help='')

    return parser.parse_args()


class BertDataIterator(Dataset):
    def __init__(self, data: Dict, max_item: int, mask_prob: float, mask_id: int, pad_id: int,
                 device=torch.device('cpu')):
        self.data = data
        self.max_item = max_item
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.device = device

    def __len__(self):
        return len(self.data)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def dynamic_masking(self, seq_ids: List) -> Tuple[Tensor, Tensor, Tensor]:
        seq_length = len(seq_ids)
        seq_tensor = self._to_tensor(seq_ids)
        label = torch.zeros(seq_length, device=self.device, dtype=torch.int64)
        # https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/albert#transformers.AlbertModel
        attention_mask = torch.ones(seq_length, device=self.device)

        for i in range(len(seq_ids)):
            if seq_ids[i] == self.pad_id:
                attention_mask[i] = 0

            prob = np.random.random()
            if self.mask_prob >= prob:
                prob /= self.mask_prob
                if prob > 0.9:  # keep
                    pass
                elif prob > 0.8:  # random id
                    seq_tensor[i] = np.random.choice(self.max_item)
                else:  # mask
                    seq_tensor[i] = self.mask_id

                label[i] = seq_ids[i]

        return seq_tensor, attention_mask, label

    def __getitem__(self, user: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.dynamic_masking(self.data[user])


class BertTestDataIterator(Dataset):
    def __init__(self, data: Dict, max_item: int, mask_prob: float, mask_id: int, pad_id: int,
                 device=torch.device('cpu')):
        self.data = data
        self.max_item = max_item
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.device = device

    def __len__(self):
        return len(self.data)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def _get_attention_mask(self, seq_ids: List):
        seq_length = len(seq_ids)
        seq_tensor = self._to_tensor(seq_ids)
        attention_mask = torch.ones(seq_length, device=self.device, dtype=torch.int32)

        for i in range(len(seq_ids)):
            if seq[i] == self.pad_id:
                attention_mask[i] = 0

        return seq_tensor, attention_mask

    def __getitem__(self, user: int) -> Tuple[Tensor, Tensor]:
        sequence = self.data[user][1:] + [self.mask_id]
        return self._get_attention_mask(sequence)


def train_progressbar(total: int, i: int, bar_length: int = 50, prefix: str = '', suffix: str = '') -> None:
    """progressbar
    """
    dot_num = int((i + 1) / total * bar_length)
    dot = '■' * dot_num
    empty = ' ' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')


def aggregate_items(df, pad_mask_id, length):
    """ aggregate user's items """
    df = df.groupby('user_id')['item_id'].agg(lambda x: x.tolist())

    for idx in df.index:
        df[idx] = df[idx][:length]
        mask_len = length - len(df[idx])
        df[idx] = [pad_mask_id for _ in range(mask_len)] + df[idx]

    return df.to_dict()


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    test_data = pd.read_csv(os.path.join(save_dir, 'test.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')

    pad_token_id = item_meta.loc[item_meta['Title'] == '[PAD]', 'item_id'].item()
    mask_token_id = item_meta.loc[item_meta['Title'] == '[MASK]', 'item_id'].item()
    num_user = int(user_meta.user_id.max())
    num_item = int(item_meta.item_id.max())
    max_len = 80

    train_data = aggregate_items(train_data, pad_mask_id=pad_token_id, length=max_len)
    iterator = BertDataIterator(train_data, max_item=num_item, mask_prob=0.2, mask_id=mask_token_id,
                                pad_id=pad_token_id)
    train_dataloader = DataLoader(iterator, batch_size=4, shuffle=True)
    test_iterator = BertTestDataIterator(train_data, max_item=num_item, mask_prob=0.2, mask_id=mask_token_id,
                                         pad_id=pad_token_id)
    test_dataloader = DataLoader(test_iterator, batch_size=32)

    model = Albert4Rec(num_item + 1, max_len)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)  # label 이 0인 경우 무시
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

    epoch = 10
    for e in range(epoch):
        # ------ train --------
        model.train()

        start_epoch_time = time.time()
        train_loss = 0
        total_step = len(train_dataloader)

        history = {}

        for step, (seq, attention, label) in enumerate(train_dataloader):
            # ------ step start ------
            if ((step + 1) % 50 == 0) | (step + 1 >= total_step):
                train_progressbar(
                    total_step, step + 1, bar_length=30,
                    prefix=f'train {e + 1:03d}/{epoch} epoch', suffix=f'{time.time() - start_epoch_time:0.2f} sec '
                )
            pred = model(seq=seq, attention=attention)
            loss = loss_func(pred, label.reshape(-1))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.zero_grad()

            train_loss += loss.item()

            if step >= total_step:
                break
            # ------ step end ------

        history['epoch'] = e + 1
        history['time'] = np.round(time.time() - start_epoch_time, 2)
        history['train_loss'] = train_loss / total_step

        sys.stdout.write(f"loss : {history['train_loss']:3.3f}")
