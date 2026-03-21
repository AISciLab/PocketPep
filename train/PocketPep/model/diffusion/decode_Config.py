import torch

class Config:
    # 数据参数
    batch_size = 32
    feature_dim = 1152
    num_classes = 25

    # 模型参数
    hidden_dim = 512
    dropout_rate = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')