import argparse
import copy
import os
import sys
import time
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adagrad, Adadelta, Adam, AdamW

from config import CONFIG
from model.metrics import RecallAtK, nDCG
from model.bert4rec import Albert4Rec, bert4Rec
# from model.bert4rec2 import bert4Rec
from model.callbacks import MlflowLogger, ModelCheckPoint
from common.utils import DotDict


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['1M'], help='데이터셋', type=str)
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)
    parser.add_argument('-e', '--epoch', default=50, help='epoch', type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('-bs', '--batch_size', default=32, help='batch size', type=int)
    parser.add_argument('-mp', '--mask_prob', default=0.2, help='masking probability', type=float)
    parser.add_argument('-sl', '--seq_length', default=80, help='length of input sequence', type=int)
    
    parser.add_argument('-nh', '--num_header', default=2, help='number of header', type=int)
    parser.add_argument('-nph', '--dim_per_header', default=32, help='header dim', type=int)
    parser.add_argument('-id', '--intermediate_dim', default=256, help='intermediate dim', type=int)
    parser.add_argument('-nl', '--num_layer', default=3, help='number of BERT layer', type=int)
    parser.add_argument('-op', '--optimizer', default='AdamW', choices=['AdamW', 'Adam', 'Adagrad', 'Adadelta'],
                        help='optimizer', type=str)
    parser.add_argument('-cpu', '--cpu', action='store_true', help='')

    return parser.parse_args()


class BertDataIterator(Dataset):
    def __init__(self, data: Dict, max_len: int, max_item: int, mask_prob: float, mask_id: int, pad_id: int,
                 device=torch.device('cpu')):
        self.data = data
        self.max_len = max_len
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
        seq_ids += [self.pad_id for _ in range(self.max_len-seq_length)] # padding
        seq_tensor = self._to_tensor(seq_ids)
        label = torch.zeros(self.max_len, device=self.device, dtype=torch.int64)
        # https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/albert#transformers.AlbertModel
        attention_mask = torch.ones(self.max_len, device=self.device)

        for i in range(self.max_len):
            if seq_ids[i] == self.pad_id:
                attention_mask[i] = 0

            prob = np.random.random()
            if self.mask_prob >= prob:
                prob /= self.mask_prob
                if prob > 0.9:    # keep 10%
                    pass
                elif prob > 0.8:  # random id 10%
                    seq_tensor[i] = np.random.choice(self.max_item)
                else:             # mask 80%
                    seq_tensor[i] = self.mask_id

                label[i] = seq_ids[i]
            
            if i+1 == seq_length:
                seq_tensor[i] = self.mask_id
                label[i] = seq_ids[i]
        
        seq_tensor *= attention_mask.int()
        return seq_tensor, attention_mask, label

    def __getitem__(self, user: int) -> Tuple[Tensor, Tensor, Tensor]:
        sequence = copy.deepcopy(self.data[user])
        return self.dynamic_masking(sequence)


class BertTestDataIterator(Dataset):
    def __init__(self, data: Dict, max_len: int, mask_id: int, pad_id: int,
                 device=torch.device('cpu')):
        self.data = data
        self.max_len = max_len
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.device = device

    def __len__(self):
        return len(self.data)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def _get_attention_mask(self, seq_ids: List):
        seq_length = len(seq_ids)
        seq_ids += [self.pad_id for _ in range(self.max_len-seq_length)] # padding
        seq_tensor = self._to_tensor(seq_ids)
        attention_mask = torch.ones(self.max_len, device=self.device, dtype=torch.int32)

        for i in range(self.max_len):
            if seq_ids[i] == self.pad_id:
                attention_mask[i] = 0

        return seq_tensor, attention_mask

    def __getitem__(self, user: int) -> Tuple[Tensor, Tensor]:
        sequence = self.data[user]['context'][1:] + [self.mask_id]
        seq_tensor, attention_mask = self._get_attention_mask(sequence)
        negative_sample = self._to_tensor(self.data[user]['negative_sample'])
        return seq_tensor, attention_mask , negative_sample


def train_progressbar(total: int, i: int, bar_length: int = 50, prefix: str = '', suffix: str = '') -> None:
    """progressbar
    """
    dot_num = int(i / total * bar_length)
    dot = '■' * dot_num
    empty = ' ' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')


def aggregate_items(df, pad_id, length):
    """ aggregate user's items """
    df.sort_values(['user_id', 'Timestamp'], inplace=True)
    df = df.groupby('user_id')['item_id'].agg(lambda x: x.tolist())

    for idx in df.index:
        df[idx] = df[idx][-length:]
        # mask_len = length - len(df[idx])
        # df[idx] = df[idx] + [pad_id for _ in range(mask_len)]

    return df.to_dict()


def get_optimizer(model, name: str, lr: float, wd: float = 0.) -> Callable:
    """ get optimizer
    Args:
        model: pytorch model
        name: optimizer name
        lr: learning rate
        wd: weight_decay(l2 regulraization)

    Returns: pytorch optimizer function
    """

    functions = {
        'Adagrad': Adagrad(model.parameters(), lr=lr, eps=0.00001, weight_decay=wd),
        'Adadelta': Adadelta(model.parameters(), lr=lr, eps=1e-06, weight_decay=wd),
        'Adam': Adam(model.parameters(), lr=lr, weight_decay=wd),
        'AdamW': AdamW(model.parameters(), lr=lr)
    }
    try:
        return functions[name]
    except KeyError:
        raise ValueError(f'optimizer [{name}] not exist, available optimizer {list(functions.keys())}')


def train(model, epoch, train_dataloader, test_dataloader, loss_func, optim, k=10, metrics=[], callback=[]):
    
    for e in range(epoch):
        # ------ train --------
        model.train()

        start_epoch_time = time.time()
        train_loss = 0
        total_step = len(train_dataloader)
        bar_step_size = max((total_step // 100), 1)

        history = {}

        for step, (seq, attention, label) in enumerate(train_dataloader):
            # ------ step start ------
            if ((step + 1) % bar_step_size == 0) | (step + 1 >= total_step):
                train_progressbar(
                    total_step, step + 1, bar_length=30,
                    prefix=f'train {e + 1:03d}/{epoch} epoch', suffix=f'{time.time() - start_epoch_time:0.2f} sec '
                )
            pred = model(seq=seq, attention=attention)
            pred = pred.reshape(-1, pred.size(-1))
            label = label.reshape(-1)
            loss = loss_func(pred, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0,)
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

        # ------ test  --------
        model.eval()
        val_loss = 0
        y_pred, y_true = [], []
        with torch.no_grad():

            for step, (seq, attention, negative) in enumerate(test_dataloader):

                label = negative[:,0]
                # random shuffle
                idx = torch.randperm(negative.shape[1])
                negative = negative[:,idx]
                
                # predict
                pred = model.predict(seq=seq, attention=attention)
                loss = loss_func(pred, label)
                val_loss += loss.item()

                # get negative sample's score
                pred = torch.gather(pred, dim=1, index=negative)
                # top k index
                _, indices = torch.topk(pred, k=k)
                # indexing samples to get top k sample
                pred = torch.gather(negative, dim=1, index=indices)

                y_pred.extend(pred.cpu().tolist())
                y_true.extend(label.cpu().tolist())

        history['val_loss'] = val_loss / step
        result = f" val_loss : {history['val_loss']:3.3f}"

        for func in metrics:
            metrics_value = func(y_pred, y_true)
            history[f'{func}'] = metrics_value
            result += f' val_{func} : {metrics_value:3.3f}'

        for func in callback:
            func(model, history)

        print(result)


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, '1M')
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    test_data = pd.read_csv(os.path.join(save_dir, 'test.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')

    pad_token_id = int(item_meta.loc[item_meta['Title'] == '[PAD]', 'item_id'].item())
    mask_token_id = int(item_meta.loc[item_meta['Title'] == '[MASK]', 'item_id'].item())
    num_user = int(user_meta.user_id.max())
    num_item = int(item_meta.item_id.max())
    
    params = DotDict({
        'learningRate': argument.learning_rate,
        'loss': 'CrossEntropyLoss',
        'maxLength': argument.seq_length,
        'optimizer': argument.optimizer,
        'k': argument.eval_k,
        'batchSize': argument.batch_size,
        'maskProb': argument.mask_prob,
        'numHeader': argument.num_header,
        'dimPerHeader': argument.dim_per_header,
        'intermediateDim': argument.intermediate_dim,
        'numLayer': argument.num_layer,
        'num_users': num_user, 'num_items': num_item
    })
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if argument.cpu:
        device = torch.device('cpu')
    
    train_set = aggregate_items(train_data, pad_id=pad_token_id, length=params.maxLength)
    iterator = BertDataIterator(train_set, max_len=params.maxLength, max_item=num_item, 
                                mask_prob=params.maskProb, mask_id=mask_token_id,
                                pad_id=pad_token_id, device=device)
    train_dataloader = DataLoader(iterator, batch_size=params.batchSize, shuffle=True, pin_memory=False)
    
    test_set = {}
    for _, row in test_data.iterrows():
        test_set[row['user_id']] = {
            'context': train_set[row['user_id']], 'negative_sample': eval(row['negative_sample'])
        }
    test_iterator = BertTestDataIterator(test_set, max_len=params.maxLength,
                                         mask_id=mask_token_id, pad_id=pad_token_id, device=device)
    test_dataloader = DataLoader(test_iterator, batch_size=params.batchSize, pin_memory=False)
    
    # HuggingFace
    n_header = params.numHeader
    dim_header = params.dimPerHeader
    model = bert4Rec(
        num_item + 1, params.maxLength, inter_dim=params.intermediateDim, 
        hidden_size=dim_header*n_header, num_head=n_header, num_layer=params.numLayer, device=device
    )
    # for param in model.parameters():
    #     torch.nn.init.trunc_normal_(param, a=-0.02, b=0.02)

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)  # label 이 0인 경우 무시
    optim = get_optimizer(model, params.optimizer, lr=params.learningRate, wd=0.01)
    
    metrics = [nDCG(), RecallAtK()]
    model_version = f'bert4rec_v{argument.model_version}'
    callback = [
        # ModelCheckPoint(
        #     os.path.join(
        #         'result', argument.dataset, # model_version + '-e{epoch:02d}-loss{val_loss:1.3f}-acc{val_acc:1.3f}.zip',
        #         model_version + '.zip'
        #     ),
        #     monitor='val_acc', mode='max'
        # ),
        MlflowLogger(experiment_name=argument.dataset, model_params=mp, run_name=model_version,
                   log_model=False, model_name='bert4rec', monitor='val_nDCG', mode='max')
    ]

    train(
        model, argument.epoch, train_dataloader, test_dataloader, loss_func, optim, 
        k=params.k, metrics=metrics, callback=callback
    )
