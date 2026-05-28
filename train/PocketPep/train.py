import argparse
import os
import pickle
from glob import glob
import random
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import shutil
import subprocess
import torch.nn.functional as F
import sys

from model.diffusion import create_diffusion
from model.affmonitor import Affinity_Reward
from model.PocketPep import DiT
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_reward(output):
    mean_value = output.mean(dim=0)
    return (mean_value+200)/10000

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
    mask = mask.float()  # 方便后续乘权重或broadcast

    return batch.to(device), mask.to(device)

def pad_and_mask(input_ids,embs, max_len, pad_value=1, device='cuda'):
    padded_seqs = []
    mask = []
    padded_embeddings = []
    for item,emb in zip(input_ids,embs):
        length = item.size(0)
        hidden_state = emb.size(1)
        # padding处理
        if length < max_len:
            pad = torch.full((max_len - length,), pad_value, dtype=item.dtype, device=device)
            padded = torch.cat([item, pad])
            m = torch.cat([torch.ones(length, dtype=torch.bool, device=device),
                           torch.zeros(max_len - length, dtype=torch.bool, device=device)])
            pad_emb = torch.zeros((max_len - length, hidden_state), dtype=emb.dtype, device=device)
            padded_emb = torch.cat([emb, pad_emb], dim=0)  # padding along seq_len dimension
        else:
            padded = item[:max_len]
            m = torch.ones(max_len, dtype=torch.bool, device=device)
            padded_emb = emb[:max_len]
        padded_seqs.append(padded)
        mask.append(m)
        padded_embeddings.append(padded_emb)

    input_ids = torch.stack(padded_seqs).to(device)  # shape = (batch_size, max_len)
    mask = torch.stack(mask).to(device)  # shape = (batch_size, max_len)
    padded_embeddings = torch.stack(padded_embeddings).to(device)
    return input_ids,mask,padded_embeddings

def data_prepare(ids,seqs,emb_path,pocket_path,device = 'cuda'):
    intput_ids = []
    embs = []
    pockets = []
    max_len = 0
    pocket_emb = []
    for id,seq in zip(ids,seqs):
        input_id = torch.tensor([all_amino_acid_number[aa] for aa in seq]).unsqueeze(0).to(device)
        max_len = max(max_len, len(seq))
        myemb_path = os.path.join(emb_path,id+'.pkl')
        with open(myemb_path, "rb") as f:
            data = pickle.load(f)
        emb = data['emb'][0][1:-1,:]
        mypocket_path = os.path.join(pocket_path,id+'.pkl')
        with open(mypocket_path, "rb") as f:
            data = pickle.load(f)
        pocket = data['mpnn_emb'][0]
        pocket_emb.append(pocket)
        intput_ids.append(input_id[0])
        embs.append(emb)
        pockets.append(pocket)
    pocket_batch, pocket_mask = pad_and_create_mask(pocket_emb)
    intput_ids,mask,embs = pad_and_mask(intput_ids,embs,max_len)
    return embs,pocket_batch,mask,pocket_mask

all_amino_acid_number = {'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
                         'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
                         'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
                         'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
                        'X': 24, '0': 0, '1': 1, '2': 2}
id_to_aa = {v: k for k, v in all_amino_acid_number.items()}

def main(args):
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion = create_diffusion(decoder_path=args.decoder_path,timestep_respacing="")
    train_df = pd.read_csv(args.train_csv_path)
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_batches = []
    total_size = len(train_df)
    for i in range(0, total_size, args.batch_size):
        batch = train_df.iloc[i:i + args.batch_size]
        train_batches.append(batch)
    model = DiT().to(device)
    reward_model = Affinity_Reward(hidden_size=320, pocket_hidden_size=384).to(device)
    state_dict = torch.load(args.monitor_path, map_location=device)
    reward_model.load_state_dict(state_dict)
    index = 0
    bmax = len(train_batches)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)

    for i in tqdm(range(args.steps)):
        if (index == bmax):
            index = 0
        model.train()
        ids = train_batches[index]['id']
        seqs = train_batches[index]['pp_seq']
        embs,pockets,mask,pocket_mask = data_prepare(ids,seqs,args.emb_path,args.pocket_path)
        model_kwargs = dict(y=pockets,mask=mask,pocket_mask = pocket_mask)
        t = torch.randint(0, diffusion.num_timesteps, (embs.shape[0],), device=device)
        loss_dict = diffusion.training_losses(model, embs, t, model_kwargs)
        loss = loss_dict['mse_loss']
        pred_seq = loss_dict['pred_seq']
        output = reward_model(pockets, pocket_mask, pred_seq, mask)
        monitor_loss = get_reward(output)
        loss = loss + monitor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i % args.ckpt_every == 0 and i != 0):
            folder_path = os.path.join(args.results_dir, f"PocketPep_{i}")
            os.makedirs(folder_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(folder_path, "model.pt"))
        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="res_model",
                        help='Directory to save trained models and outputs.')
    parser.add_argument("--steps", type=int, default=40001,
                        help='Total number of training steps.')
    parser.add_argument("--ckpt-every", type=int, default=2000,
                        help='Number of steps between saving checkpoints.')
    parser.add_argument("--decoder-path", type=str, default='../../datasets/decoder_Pep/model.pt',
                        help='Path to the pre-trained decoder model.')
    parser.add_argument("--monitor-path", type=str, default='../../datasets/aff_monitor/model.pt',
                        help='Path to the pre-trained monitor model.')
    parser.add_argument("--pocket-path", type=str, default='../datasets/pocket_emb',
                        help='Path to pocket structure data.')
    parser.add_argument("--emb-path", type=str, default='../datasets/pp_emb',
                        help='Path to peptide seqence data.')
    parser.add_argument("--train-csv-path", type=str, default='../datasets/train.csv',
                        help='Path to the training dataset (CSV file).')
    parser.add_argument("--batch-size", type=int, default=32,
                        help='Batch size for training.')
    args = parser.parse_args()
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main(args)
