import argparse
import os
import pickle
from glob import glob
import random

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch.nn.functional as F

from model.diffusion import create_diffusion
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

def data_prepare(pocket_path,device = 'cuda'):
    with open(pocket_path, "rb") as f:
        data = pickle.load(f)
    pocket = data['mpnn_emb'].to(device)
    mask = torch.ones(1, pocket.shape[1]).to(device)
    return pocket,mask

all_amino_acid_number = {'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
                         'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
                         'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
                         'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
                        'X': 24, '0': 0, '1': 1, '2': 2}
id_to_aa = {v: k for k, v in all_amino_acid_number.items()}

def main(args):
    set_seed()
    pt_path = os.path.join(args.ckpt_path,'model.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(pt_path, map_location=device)
    diffusion = create_diffusion(str(args.num_sampling_steps),decoder_path=args.decoder_path)
    model = DiT().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    pockets,pocket_mask = data_prepare(args.pocket_path,device)
    model_kwargs = dict(y=pockets,pocket_mask = pocket_mask)
    noise = torch.randn(1, args.length, 1152).to(device)
    res = diffusion.p_sample_loop(
        model, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
        device=device
    )
    seq_ids = res.squeeze().tolist()
    decoded_seq = "".join(id_to_aa[i] for i in seq_ids)
    fasta_content = f">peptide\n{decoded_seq}\n"
    with open(f"{args.results_dir}/peptide.fasta", "w") as f:
        f.write(fasta_content)
    print('word done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="example/output")
    parser.add_argument("--ckpt-path", type=str, default='datasets/PocketPep_ckpt')
    parser.add_argument("--decoder-path", type=str, default='datasets/decoder_Pep/model.pt')
    parser.add_argument("--pocket-path", type=str, default='example/pocket.pkl')
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    args = parser.parse_args()
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main(args)
