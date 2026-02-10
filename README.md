# Pocket-aware Conditional Latent Diffusion Model for Peptide Design
Peptides often exert therapeutic effects on diseases by binding to target proteins.
Diffusion models have emerged as a promising paradigm for peptide design.
However, most methods ignore the structural specificity of target proteins, and cannot monitor affinity during training process, thus limiting their applicability to target-aware peptide design.

Therefore, we propose a target-specific peptide design framework, called **PocketPep**, that harmonizes 3D pocket-aware conditional latent diffusion with specific feedback mechanisms.

![pipeline](./picture/fig-pipeline.png)
## 1.Environment Setup

For the denoising model, please refer to the dependencies specified in the  [`environment.yml`](environment.yml) file.  
Since ESMC has higher Python version requirements, an additional environment is needed. Please ensure you use a newer Python version (e.g., Python 3.12 or above) to properly obtain sequence embeddings.
