import argparse
import os
import pickle

import pandas as pd
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from torch.optim import AdamW

from model.reward1 import Affinity_Reward
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

amino_acid_list = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D','P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
amino_acid_to_index = {aa: idx for idx, aa in enumerate(amino_acid_list)}

def preprocess_function(pocket_path,seq,device='cuda'):
    with open(pocket_path, "rb") as f:
        data = pickle.load(f)
    pocket_emb = data['mpnn_emb'].to(device)
    L = len(seq)
    intput_ids = torch.zeros(1 ,L, 20).to(device)
    for i, aa in enumerate(seq):
        if aa in amino_acid_to_index:
            intput_ids[0 ,i, amino_acid_to_index[aa]] = 1
    pocket_mask = torch.ones(pocket_emb.shape[0], pocket_emb.shape[1], device=pocket_emb.device, dtype=torch.float32)
    pp_mask = torch.ones(intput_ids.shape[0], intput_ids.shape[1], device=intput_ids.device, dtype=torch.float32)
    res = {
        "pocket_embs_tensor": pocket_emb,
        "pocket_mask": pocket_mask,
        'pp_embs_tensor': intput_ids,
        'pp_mask': pp_mask,
    }
    return res

def main(args):
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Affinity_Reward(hidden_size=320, pocket_hidden_size=384).to(device)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        data = preprocess_function(args.pocket_path,args.seq,device)
        output = model(data['pocket_embs_tensor'], data['pocket_mask'], data['pp_embs_tensor'], data['pp_mask'])
    score = output.item()
    print('Aff is ',score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default='datasets/aff_monitor/model.pt')
    parser.add_argument("--seq", type=str, default='HFTVWHDYSI')
    parser.add_argument("--pocket-path", type=str, default='example/pocket.pkl')
    args = parser.parse_args()
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main(args)


