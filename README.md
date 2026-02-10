# Pocket-aware Conditional Latent Diffusion Model for Peptide Design
Peptides often exert therapeutic effects on diseases by binding to target proteins.
Diffusion models have emerged as a promising paradigm for peptide design.
However, most methods ignore the structural specificity of target proteins, and cannot monitor affinity during training process, thus limiting their applicability to target-aware peptide design.

Therefore, we propose a target-specific peptide design framework, called **PocketPep**, that harmonizes 3D pocket-aware conditional latent diffusion with specific feedback mechanisms.

![pipeline](./picture/fig-pipeline.png)
## 1.Environment Setup

For the denoising model, please refer to the dependencies specified in the  [`./environment.yml`](environment.yml) file.  
Since ESMC has higher Python version requirements, an additional environment is needed. Please ensure you use a newer Python version (e.g., Python 3.12 or above) to properly obtain sequence embeddings.

The model weights can be downloaded from [this link](https://drive.google.com/file/d/1SuhqbCUKjTJS0Fp-YkQsx5OGIgJW_PWX/view?usp=drive_link).
Please download them and replace the corresponding directory with the downloaded files.

## 2.Example Workflow
**First**, structural features of the target binding pocket are extracted using `get_pocket.py` from the `ProteinMPNN` module in the `datasets` directory. 

These pocket representations are subsequently used to guide the generation of compatible peptide sequences and to support downstream binding affinity prediction.

You can view the full list of command-line arguments and their descriptions for any Python script by running it with the `--help` flag. For example:

```bash
python get_pocket.py --pdb-path ../../example/pocket.pdb --out-path ../../example/pocket.pkl
```

**Second**, you can generate peptide sequences conditioned on a precomputed pocket structural feature file (`.pkl`). 
You are free to specify the peptide length (recommended range: 5–50 residues) and the number of denoising sampling steps (recommended: 50).For example:
```bash
python sample_generate_seq.py \
  --out-path example/output/peptide.fasta \
  --ckpt-path datasets/PocketPep_ckpt \
  --decoder-path datasets/decoder_Pep/model.pt \
  --pocket-path example/pocket.pkl \
  --length 10 \
  --num-sampling-steps 50
```
This will display the following help message:
```text
usage: sample_generate_seq.py [-h] [--out-path OUT_PATH] [--ckpt-path CKPT_PATH]
                              [--decoder-path DECODER_PATH] [--pocket-path POCKET_PATH]
                              [--length LENGTH] [--num-sampling-steps NUM_SAMPLING_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --out-path OUT_PATH   Path to the output FASTA file where the extracted protein sequence will be saved.
  --ckpt-path CKPT_PATH
                        Path to the PocketPep model checkpoint file.
  --decoder-path DECODER_PATH
                        Path to the Decoder model checkpoint file.
  --pocket-path POCKET_PATH
                        Path to the PKL file containing the structural features of the peptide-binding pocket.
  --length LENGTH       Length of the peptide sequence to be generated.
  --num-sampling-steps NUM_SAMPLING_STEPS
                        Number of denoising sampling steps used during peptide sequence generation.
```
**Third**, you can compute the binding affinity scores between the **peptide sequences generated in Step 2** and the **pocket structural features extracted in Step 1**. 

These predicted affinity scores can be used for downstream filtering, ranking, or experimental validation of high-potential peptide candidates.For example:
```bash
python sample_predict_aff.py \
  --seq "HFTVWHDYSI" \
  --pocket-path example/pocket.pkl \
  --ckpt-path datasets/aff_monitor/model.pt \
  --out-path example/output/res.csv
```
This will display the following help message:
```text
usage: sample_predict_aff.py [-h] [--out-path OUT_PATH] [--ckpt-path CKPT_PATH] [--seq SEQ] [--pocket-path POCKET_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --out-path OUT_PATH   Path to the output file where the predicted affinity scores will be saved.
  --ckpt-path CKPT_PATH
                        Path to the aff_monitor model checkpoint file.
  --seq SEQ             Peptide sequence used for inferring binding affinity.
  --pocket-path POCKET_PATH
                        Structural representation of the binding pocket used for inferring peptide affinity.
```


