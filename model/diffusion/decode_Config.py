import torch

class Config:
    # 数据参数
    batch_size = 32
    feature_dim = 1152  # 每个特征的维度
    num_classes = 25  # 氨基酸类别数

    # 模型参数
    hidden_dim = 512  # 隐藏层维度
    dropout_rate = 0.0  # Dropout概率

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')