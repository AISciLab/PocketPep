import argparse
import os
import pickle

import pandas as pd
import torch
import random
import numpy as np
from torch.optim import AdamW

from model.ESMC_decoder import ESMC_decoder
import torch.nn as nn

from tqdm import tqdm


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
all_amino_acid_number = {'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
                         'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
                         'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
                         'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
                        'X': 24, '0': 0, '1': 1, '2': 2}


def get_emb(ids,max_len,pkl_path = "../datasets/pp_emb"):
    arr = []
    for id in ids:
        file_path = os.path.join(pkl_path,f'{id}.pkl')
        with open(file_path,'rb') as f:
            data = pickle.load(f)
        emb = data['emb'][0][1:-1]
        seq_len = emb.shape[0]
        pad_len = max_len - seq_len
        padding = torch.zeros(pad_len, 1152).to(device)
        padded_emb = torch.cat([emb, padding], dim=0)
        arr.append(padded_emb)
    return torch.stack(arr)


def esm_encoder_seq(seq, max_len):
    seqlen = len(seq)
    # 编码序列
    s = [all_amino_acid_number[x] for x in seq]

    # mask: 1 表示真实 token
    mask = [1] * seqlen

    # 如果序列长度小于 max_len，则 padding
    if seqlen < max_len:
        pad_len = max_len - seqlen
        s += [all_amino_acid_number['1']] * pad_len
        mask += [0] * pad_len
    else:
        # 超长截断
        s = s[:max_len]
        mask = mask[:max_len]
    return torch.tensor(s), torch.tensor(mask)

def preprocess_function(samples):
    processed_samples = {
        "input_ids": [],
        "mask": [],
        'seqs': [],
        'ids':[]
    }
    max_len = samples["pp_seq"].str.len().max()
    for id,seq in zip(samples['id'],samples['pp_seq']):
        input_ids, mask = esm_encoder_seq(seq, max_len)
        processed_samples['input_ids'].append(input_ids)
        processed_samples['mask'].append(mask)
        processed_samples['seqs'].append(seq)
        processed_samples['ids'].append(id)
    res = {
        "labels": torch.stack(processed_samples['input_ids']),
        "masks": torch.stack(processed_samples['mask']),
        'seqs': processed_samples['seqs'],
        'ids': processed_samples['ids']
    }
    return res

def data_prepare(train_csv_path,test_csv_path,batch_size):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_batches = []
    test_batches = []
    total_size = len(train_df)
    test_size = len(test_df)
    for i in range(0, total_size, batch_size):
        batch = train_df.iloc[i:i + batch_size]
        batch = preprocess_function(batch)
        train_batches.append(batch)

    for i in range(0, test_size, batch_size):
        batch = test_df.iloc[i:i + batch_size]
        batch = preprocess_function(batch)
        test_batches.append(batch)
    return train_batches, test_batches

def train(n_steps,batch_size,learning_rate,min_learning_rate,train_csv_path,test_csv_path,checkpoint_steps,output_path):
    model = ESMC_decoder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=min_learning_rate)
    train_batches, test_batches = data_prepare(train_csv_path, test_csv_path, batch_size)
    best_acc = 0.0
    idx = 0
    bmax = len(train_batches)

    for i in tqdm(range(n_steps)):
        if (idx == bmax):
            idx = 0
        model.train()
        masks = train_batches[idx]["masks"].to(device)
        labels = train_batches[idx]["labels"].to(device)
        ids = train_batches[idx]['ids']
        max_len = masks.shape[1]
        emb = get_emb(ids, max_len)
        outputs = model(emb, masks)  # [B, L, C]
        outputs_reshaped = outputs.reshape(-1, outputs.size(-1))  # [B*L, C]
        labels_reshaped = labels.reshape(-1)  # [B*L]
        mask_reshaped = masks.reshape(-1).bool()  # [B*L]
        valid_outputs = outputs_reshaped[mask_reshaped]
        valid_labels = labels_reshaped[mask_reshaped]
        loss = criterion(valid_outputs, valid_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        idx = idx + 1
        if (i % checkpoint_steps == 0 and i != 0):
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for item in test_batches:
                    masks = item["masks"].to(device)
                    label = item["labels"].to(device)
                    ids = item["ids"]
                    max_len = masks.shape[1]
                    emb = get_emb(ids, max_len)
                    outputs = model(emb, masks)
                    preds = torch.argmax(outputs, dim=-1)
                    valid_preds = preds[masks == 1]
                    valid_labels = label[masks == 1]
                    correct += (valid_preds == valid_labels).sum().item()
                    total += valid_labels.numel()
                acc = correct / total

            if (acc > best_acc):
                best_acc = acc
                os.makedirs(output_path, exist_ok=True)
                torch.save(model.state_dict(), f'{output_path}/esm_classifier.pt')

if __name__ == "__main__":

    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=3001,help='Total number of training steps.')
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=32, help='Batch size for training.')
    parser.add_argument('--learning-rate', action='store', dest='learning_rate', type=float,
                        default=1e-3, help='Initial learning rate.')
    parser.add_argument('--min-learning-rate', action='store', dest='min_learning_rate', type=float,
                        default=5e-5, help='Minimum learning rate for cosine annealing.')
    parser.add_argument('--train', action='store', dest='train_csv_path',
                        default='../datasets/train.csv',help='Path to the training dataset (CSV file).')
    parser.add_argument('--test', action='store', dest='test_csv_path',
                        default='../datasets/test.csv', help='Path to the testing dataset (CSV file).')
    parser.add_argument('--checkpoint-steps', action='store', dest='checkpoint_steps', type=int,
                        default=200,help='Number of steps between saving checkpoints.')
    parser.add_argument('--output', action='store', dest='output_path',
                        default='res_model', help='Directory to save trained models and outputs.')
    arg_dict = vars(parser.parse_args())
    train(**arg_dict)
