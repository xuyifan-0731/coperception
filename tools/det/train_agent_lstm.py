import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='/data/V2X-Sim-det/train/agent1')
parser.add_argument('--val_path', type=str, default='/data/V2X-Sim-det/val/agent1')
parser.add_argument('--test_path', type=str, default='/data/V2X-Sim-det/test/agent1')
parser.add_argument('--save_model_path', type=str, default='/data/V2X-Sim-pre-prediction/agent1')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--input_len', type=int, default=10)  # 输入帧数
parser.add_argument('--delay_step', type=int, default=1)  # 延迟t对应的帧数（如0.2s=1）
args = parser.parse_args()


# --------------------------- 数据加载 ---------------------------
class V2XDataset(Dataset):
    def __init__(self, root, input_len, delay_step):
        self.root = root
        self.input_len = input_len
        self.delay_step = delay_step
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        scene_ids = sorted(set(f.split('_')[0] for f in os.listdir(self.root)))
        for scene_id in scene_ids:
            for t in range(0, 100 - self.input_len - self.delay_step):
                input_files = [os.path.join(self.root, f'{scene_id}_{t+i}', '0.npy') for i in range(self.input_len)]
                target_file = os.path.join(self.root, f'{scene_id}_{t+self.input_len+self.delay_step-1}', '0.npy')
                samples.append((input_files, target_file))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_files, target_file = self.samples[idx]
        input_seq = []
        for file in input_files:
            data = np.load(file, allow_pickle=True).item()
            reg = data['reg_target_sparse']
            reg = np.mean(reg, axis=0).squeeze()  # 均值特征向量
            input_seq.append(reg)

        target_data = np.load(target_file, allow_pickle=True).item()
        target = np.mean(target_data['reg_target_sparse'], axis=0).squeeze()
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# --------------------------- LSTM 网络 ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])  # 取最后一步输出


# --------------------------- 训练与评估 ---------------------------
def train_and_eval():
    # 数据
    train_ds = V2XDataset(args.train_path, args.input_len, args.delay_step)
    val_ds = V2XDataset(args.val_path, args.input_len, args.delay_step)
    test_ds = V2XDataset(args.test_path, args.input_len, args.delay_step)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 模型
    model = LSTMModel()
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                val_loss += criterion(pred, y).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # 模型保存
    os.makedirs(args.save_model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_model_path, 'lstm_model.pth'))

    # 损失图
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(os.path.join(args.save_model_path, 'loss_curve.png'))

    # 测试可视化
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.cuda()
            pred = model(x).cpu().numpy()
            preds.append(pred)
            gts.append(y.numpy())

    preds = np.array(preds).squeeze()
    gts = np.array(gts).squeeze()
    for i in range(preds.shape[1]):
        plt.figure()
        plt.plot(preds[:, i], label='Pred')
        plt.plot(gts[:, i], label='GT')
        plt.title(f"Feature {i} Prediction")
        plt.legend()
        plt.savefig(os.path.join(args.save_model_path, f'feature_{i}_prediction.png'))


if __name__ == '__main__':
    train_and_eval()
