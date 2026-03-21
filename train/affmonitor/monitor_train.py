import argparse
import os
import pickle

import pandas as pd
import torch
import random
import numpy as np
from torch.optim import AdamW

from model.affmonitor import Affinity_Reward
import torch.nn as nn
import torch.nn.functional as F
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

amino_acid_list = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D','P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
amino_acid_to_index = {aa: idx for idx, aa in enumerate(amino_acid_list)}

with open('../datasets/pocket_score.pkl', 'rb') as pkl_file:
    train_pocket_score = pickle.load(pkl_file)
with open('../datasets/receptor_score.pkl', 'rb') as pkl_file:
    train_receptor_score = pickle.load(pkl_file)

def pad_and_create_mask(tensor_list, max_len=None, pad_value=0.0, device='cuda'):
    lengths = [x.size(0) for x in tensor_list]
    if max_len is None:
        max_len = max(lengths)
    padded_tensors = []
    for x in tensor_list:
        pad_len = max_len - x.size(0)
        padded_x = F.pad(x, (0, 0, 0, pad_len), value=pad_value)
        padded_tensors.append(padded_x)

    batch = torch.stack(padded_tensors, dim=0)  # [N, max_L, C]

    mask = torch.arange(max_len, device=batch.device)[None, :] < torch.tensor(lengths, device=batch.device)[:, None]
    mask = mask.float()
    return batch.to(device), mask.to(device)

def read_fasta_to_dict(file_path):
    fasta_dict = {}
    with open(file_path, 'r') as file:
        seq_list = []
        current_id = None
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    fasta_dict[current_id] = ''.join(seq_list)
                current_id = line[1:]
                seq_list = []
            else:
                seq_list.append(line)
        if current_id:
            fasta_dict[current_id] = ''.join(seq_list)
    return fasta_dict

def get_loss(outputs,scores):
    criterion = nn.MSELoss()
    loss1 = torch.sqrt(criterion(outputs, scores))
    loss1 = loss1.float()
    loss = loss1
    return loss

def preprocess_function(samples,receptor_path,pocket_path,pp_path = '../datasets/pp_fasta'):
    ids = []
    pocket_embs = []
    pp_embs = []
    scores= [ ]
    pocket_maxlen = 0
    pp_maxlen = 0
    for id in samples['id']:
        ids.append(id)
        if id.startswith('pocket'):
            id = id[7:]
            pocket = os.path.join(pocket_path,f'{id}.pkl')
            with open(pocket, "rb") as f:
                data = pickle.load(f)
            pocket_emb = data['mpnn_emb'][0]
            pocket_maxlen = max(pocket_maxlen, len(pocket_emb))
            pp_fasta = os.path.join(pp_path,f'{id}.fasta')
            pp_dict = read_fasta_to_dict(pp_fasta)
            scores.append(train_pocket_score[id]['score'][0])
            pocket_embs.append(pocket_emb)
            iid = f'{id}_0'
            seq = pp_dict[iid]
            pp_maxlen = max(pp_maxlen, len(seq))
            L = len(seq)
            tensor = torch.zeros(L, 20)
            for i, aa in enumerate(seq):
                if aa in amino_acid_to_index:
                    tensor[i, amino_acid_to_index[aa]] = 1
            pp_embs.append(tensor)
        else:
            id = id[9:]
            scores.append(train_receptor_score[id]['score'][0])
            receptor = os.path.join(receptor_path, f'{id}.pkl')
            with open(receptor, "rb") as f:
                data = pickle.load(f)
            receptor_emb = data['mpnn_emb'][0]
            pocket_embs.append(receptor_emb)
            pocket_maxlen = max(pocket_maxlen, len(receptor_emb))
            pp_fasta = os.path.join(pp_path, f'r_{id}.fasta')
            pp_dict = read_fasta_to_dict(pp_fasta)
            seq = pp_dict[id]
            pp_maxlen = max(pp_maxlen, len(seq))
            L = len(seq)  # 序列的长度
            tensor = torch.zeros(L, 20)
            for i, aa in enumerate(seq):
                if aa in amino_acid_to_index:
                    tensor[i, amino_acid_to_index[aa]] = 1
            pp_embs.append(tensor)
    pocket_embs_tensor,pocket_mask = pad_and_create_mask(pocket_embs, max_len=pocket_maxlen)
    pp_embs_tensor,pp_mask = pad_and_create_mask(pp_embs, max_len=pp_maxlen)
    scores_tensor = torch.tensor(scores).unsqueeze(1).to(device)
    res = {
        "pocket_embs_tensor": pocket_embs_tensor,
        "pocket_mask": pocket_mask,
        'pp_embs_tensor': pp_embs_tensor,
        'pp_mask': pp_mask,
        'scores': scores_tensor
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
        train_batches.append(batch)
    for i in range(0, test_size, batch_size):
        batch = test_df.iloc[i:i + batch_size]
        test_batches.append(batch)
    return train_batches, test_batches



def train(n_steps,batch_size,learning_rate,receptor_path,pocket_path,train_csv_path,test_csv_path,checkpoint_steps,output_path):
    model = Affinity_Reward(hidden_size=320, pocket_hidden_size=384).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=learning_rate)
    train_batches, test_batches = data_prepare(train_csv_path, test_csv_path, batch_size)
    idx = 0
    bmax = len(train_batches)

    for i in tqdm(range(n_steps)):
        if (idx == bmax):
            idx = 0
        model.train()
        batch = train_batches[idx]
        data = preprocess_function(batch,receptor_path,pocket_path)
        output = model(data['pocket_embs_tensor'], data['pocket_mask'], data['pp_embs_tensor'], data['pp_mask'])
        loss = get_loss(output, data['scores'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        idx = idx + 1
        if (i % checkpoint_steps == 0 and i!=0):
            os.makedirs(output_path, exist_ok=True)
            torch.save(model.state_dict(), f'{output_path}/affmonitor_{i}.pt')


if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=40001,help='Total number of training steps.')
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=64, help='Batch size for training.')
    parser.add_argument('--learning-rate', action='store', dest='learning_rate', type=float,
                        default=5e-5, help='Initial learning rate.')
    parser.add_argument('--receptor', action='store', dest='receptor_path',
                        default='../datasets/receptor_emb', help='Path to receptor structure data.')
    parser.add_argument('--pocket', action='store', dest='pocket_path',
                        default='../datasets/pocket_emb', help='Path to pocket structure data.')
    parser.add_argument('--train', action='store', dest='train_csv_path',
                        default='../datasets/monitor_train.csv',help='Path to the training dataset (CSV file).')
    parser.add_argument('--test', action='store', dest='test_csv_path',
                        default='../datasets/monitor_test.csv', help='Path to the testing dataset (CSV file).')
    parser.add_argument('--checkpoint-steps', action='store', dest='checkpoint_steps', type=int,
                        default=200,help='Number of steps between saving checkpoints.')
    parser.add_argument('--output', action='store', dest='output_path',
                        default='res_model', help='Directory to save trained models and outputs.')
    arg_dict = vars(parser.parse_args())
    train(**arg_dict)
