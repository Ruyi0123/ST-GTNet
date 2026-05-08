import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class GRUTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nheads, n_layers=2):
        super().__init__()
        # GRU模块捕捉时间依赖[2,4](@ref)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Transformer编码器捕捉特征关联[8](@ref)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 空间位置编码（经纬度特征增强）
        self.pos_encoder = nn.Linear(2, hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x, locations):
        # GRU处理时间序列
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_dim]
        
        # 取最后一个时间步[5](@ref)
        last_step = gru_out[:, -1, :]  # [batch, hidden_dim]
        
        # 位置特征融合
        loc_embed = self.pos_encoder(locations)  # [batch, hidden_dim]
        combined = last_step + loc_embed
        
        # Transformer特征增强
        transformer_in = combined.unsqueeze(1)  # [batch, 1, hidden_dim]
        transformer_out = self.transformer_encoder(transformer_in)  # [batch, 1, hidden_dim]
        
        # 最终预测
        return self.fc(transformer_out.squeeze(1))

# 模型结构
class STLSTM(nn.Module):
    def __init__(self, input_size, spatial_size=16, hidden_size=64, num_layers=2):
        super().__init__()
        # 空间特征编码（经纬度→高维嵌入）
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, spatial_size),
            nn.ReLU()
        )
        # 时序特征编码（LSTM处理年度指标）
        self.lstm = nn.LSTM(
            input_size=input_size + spatial_size,  # 拼接指标和空间特征
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # 回归输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出CO2排放量
        )

    def forward(self, x_temporal, x_spatial):
        # x_temporal: (batch, seq_len, feature_num)
        # x_spatial: (batch, 2) 经纬度
        spatial_emb = self.spatial_embed(x_spatial)  # (batch, spatial_size)
        spatial_emb = spatial_emb.unsqueeze(1).repeat(1, x_temporal.size(1), 1)  # 复制到每个时间步
        combined = torch.cat([x_temporal, spatial_emb], dim=-1)  # 拼接特征
        lstm_out, _ = self.lstm(combined)  # (batch, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # 取最后一个时间步
        return self.fc(last_output).squeeze(-1)  # (batch,)