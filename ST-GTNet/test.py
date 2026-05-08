import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. 数据预处理模块（增加R²记录）
class EmissionDataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)
        self.features = [
            '耕地聚集指数', '耕地比例指数', '土地利用多样性指数', '建设用地均斑',
            '生态用地比例', 'ESV指数', '土地扩张强度', '城市化进程指数',
            '耕地转入转出强度', '荒地退化速率', '耕地LPI', '经纬度'
        ]
        self.target = '二氧化碳排放量'
        
    def handle_missing_data(self):
        # ...（保持不变）...
        return self.df

    def preprocess(self):
        # ...（保持不变）...
        return np.array(sequences, dtype=object)

# 2. 神经网络模型架构（保持不变）
class GRUTransformerModel(nn.Module):
    # ...（保持不变）...

# 3. 数据集类（保持不变）
class EmissionDataset(Dataset):
    # ...（保持不变）...

# 4. 增强的训练流程（增加R²计算和可视化）
def main():
    # 数据准备
    processor = EmissionDataProcessor("co2_data.xlsx")
    sequences = processor.preprocess()
    train_data, test_data = train_test_split(sequences, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_loader = DataLoader(EmissionDataset(train_data), batch_size=32, shuffle=True)
    test_loader = DataLoader(EmissionDataset(test_data), batch_size=32)
    
    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUTransformerModel(
        input_dim=len(processor.features)-2,
        hidden_dim=64,
        output_dim=1,
        nheads=4
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # ===== 新增：训练指标记录 =====
    train_losses, val_losses = [], []
    train_r2_scores, val_r2_scores = [], []
    best_val_r2 = -np.inf  # R²初始化为负无穷
    
    # 训练循环（增加R²计算）
    for epoch in range(100):
        model.train()
        epoch_train_loss = 0
        train_preds, train_targets = [], []
        
        # 训练阶段
        for features, locs, targets in train_loader:
            features, locs, targets = features.to(device), locs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, locs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            # 收集训练预测和目标
            train_preds.append(outputs.detach().cpu())
            train_targets.append(targets.detach().cpu())
        
        # 计算训练集R²
        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_r2 = r2_score(train_targets, train_preds)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for features, locs, targets in test_loader:
                features, locs, targets = features.to(device), locs.to(device), targets.to(device)
                outputs = model(features, locs)
                loss = criterion(outputs, targets)
                
                epoch_val_loss += loss.item()
                # 收集验证预测和目标
                val_preds.append(outputs.cpu())
                val_targets.append(targets.cpu())
        
        # 计算验证集R²
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        # 记录指标
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(test_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)
        
        # 更新学习率和提前停止逻辑
        scheduler.step(val_loss)
        
        # 保存最佳R²模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), "best_co2_model.pth")
        
        # 每5个epoch打印指标
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")
    
    # ===== 新增：可视化指标 =====
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # R²曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_r2_scores, label='Train R²')
    plt.plot(val_r2_scores, label='Val R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Training & Validation R² Scores')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # 参考线
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    print(f"Best Validation R²: {best_val_r2:.4f}")

if __name__ == "__main__":
    main()