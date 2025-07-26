import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("Device Count:", torch.cuda.device_count())
print("Memory Allocated:", torch.cuda.memory_allocated())
print("Memory Cached:", torch.cuda.memory_reserved())

# ========== CONFIG ==========
DATA_ROOT = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-det/train/agent1"
SAVE_PATH = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-com-pred-model/agent1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_SAMPLES = 5
Batch_size = 16
EPOCH=20
FIELDS = {
    'reg_target_sparse': 6,
    'voxel_indices_0': 3,
    'gt_max_iou': 2,
    'label_sparse': 1,
    'allocation_mask': 1,
    'reg_loss_mask': 1,
    'vis_occupy_indices': 3,
    'vis_free_indices': 3,
    'target_agent_id': 1,
    'num_sensor': 1,
    'trans_matrices': 16,
    'trans_matrices_no_cross_road': 16,
    'voxel_indices_teacher': 3,
    'voxel_indices_teacher_no_cross_road': 3
}

scalers = {
    'gt_max_iou': MinMaxScaler(),
    'vis_occupy_indices': MinMaxScaler(),
    'vis_free_indices': MinMaxScaler(),
    'trans_matrices': MinMaxScaler(),
    'trans_matrices_no_cross_road': MinMaxScaler(),
    'reg_target_sparse': MinMaxScaler(),
    'voxel_indices_0': MinMaxScaler(),
    'voxel_indices_teacher': MinMaxScaler(),
    'voxel_indices_teacher_no_cross_road': MinMaxScaler(),
}

# ========== MODEL ==========
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=6, d_model=64, num_heads=4, num_layers=2, num_classes=None):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.delay_embed = nn.Sequential(
            nn.Embedding(10, d_model),
            nn.LayerNorm(d_model)
        )
        self.num_classes = num_classes
        output_dim = num_classes if num_classes else input_dim
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x, delay):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # last token
        d = self.delay_embed(delay).squeeze(1)
        h = torch.cat([x, d], dim=1)
        return self.output_head(h)

# ========== DATASET ==========
class V2XDataset(Dataset):
    def __init__(self, root_dir, key, frame_len=100):
        self.root = root_dir
        self.key = key
        self.input_dim = FIELDS[key]
        self.samples = []

        # 自动提取实际存在的 scene id
        scene_ids = sorted(set(
            int(name.split('_')[0])
            for name in os.listdir(root_dir)
            if '_' in name and os.path.isdir(os.path.join(root_dir, name))
        ))

        for scene in scene_ids:
            for frame in range(frame_len):
                self.samples.append((scene, frame))

    def extract_feature(self, data):
        val = data[self.key]
        if self.key == 'reg_target_sparse':
            return val.mean(axis=0).squeeze()
        elif self.key in ['voxel_indices_0', 'voxel_indices_teacher', 'voxel_indices_teacher_no_cross_road']:
            return val.mean(axis=0)
        elif self.key == 'gt_max_iou':
            return val[:, :2].mean(axis=0)
        elif self.key in ['label_sparse', 'allocation_mask', 'reg_loss_mask']:
            return np.array([val.mean()])
        elif self.key in ['vis_occupy_indices', 'vis_free_indices']:
            return val[:3, :].mean(axis=1)
        elif self.key == 'target_agent_id':
            return np.array([val])
        elif self.key == 'num_sensor':
            return np.array([int(val)])
        elif self.key in ['trans_matrices', 'trans_matrices_no_cross_road']:
            val = val.reshape(-1, 16)
            return val.mean(axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, frame = self.samples[idx]
        past_k = np.random.randint(1, 6)
        delays = np.random.choice([1, 2, 3])
        frame_ids = sorted(np.random.choice(range(100), past_k, replace=False))

        max_len = 5
        feats = []
        for f in frame_ids:
            folder = os.path.join(self.root, f"{scene}_{f}")
            file_path = os.path.join(folder, "0.npy")
            if not os.path.exists(file_path):
                continue
            try:
                data = np.load(file_path, allow_pickle=True).item()
                feat = self.extract_feature(data)
                if self.key in scalers:
                    feat = scalers[self.key].fit_transform(feat.reshape(-1, 1)).flatten()
                feats.append(feat)
            except:
                continue

        if len(feats) == 0:
            raise IndexError("No valid frames")

        if len(feats) < max_len:
            pad = [np.zeros(self.input_dim)] * (max_len - len(feats))
            feats = feats + pad
        elif len(feats) > max_len:
            feats = feats[:max_len]

        target_fid = min(frame_ids[-1] + delays, 99)
        target_path = os.path.join(self.root, f"{scene}_{target_fid}", "0.npy")
        target = np.zeros(self.input_dim)
        if os.path.exists(target_path):
            try:
                target_data = np.load(target_path, allow_pickle=True).item()
                target = self.extract_feature(target_data)
                if self.key in scalers:
                    target = scalers[self.key].transform(target.reshape(-1, 1)).flatten()
            except:
                pass

        return {
            "x": torch.tensor(np.stack(feats), dtype=torch.float32),
            "delay": torch.tensor([delays], dtype=torch.long),
            "y": torch.tensor(target, dtype=torch.float32),
        }

# ========== TRAIN WITH VALIDATION ==========
def train_field(key, input_dim):
    print(f"\n==== Training field: {key} ====")
    train_set = V2XDataset(DATA_ROOT, key)
    val_set = V2XDataset("/data/jqy/V2X-Sim-2.0/V2X-Sim-2-det/val/agent1", key)

    train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=Batch_size, shuffle=False)

    num_classes = 4 if key == 'num_sensor' else None
    model = TransformerPredictor(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_history = []
    val_loss_history = []

    for epoch in range(EPOCH):
        model.train()
        total_loss, skipped = 0, 0
        for sample in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            try:
                x = sample['x'].to(DEVICE)
                delay = sample['delay'].to(DEVICE)
                y = sample['y'].to(DEVICE)
                out = model(x, delay)

                if key in ['allocation_mask', 'reg_loss_mask']:
                    weights = (y > 0.01).float() * 5.0 + 1.0
                    loss = F.mse_loss(out, y, reduction='none')
                    loss = (loss * weights).mean()
                elif key in ['label_sparse', 'target_agent_id']:
                    loss = F.binary_cross_entropy_with_logits(out, y)
                elif key == 'num_sensor':
                    y_class = (y.view(-1).long() - 3).clamp(min=0, max=3)
                    loss = F.cross_entropy(out, y_class)
                else:
                    loss = F.mse_loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except:
                skipped += 1

        avg_loss = total_loss / (len(train_loader) - skipped + 1e-5)
        loss_history.append(avg_loss)
        if key in scalers:
            joblib.dump(scalers[key], os.path.join(SAVE_PATH, f"scaler_{key}.pkl"))

        # ==== VALIDATION ====
        model.eval()
        preds, targets = [], []
        val_loss = 0
        with torch.no_grad():
            for sample in val_loader:
                x = sample['x'].to(DEVICE)
                delay = sample['delay'].to(DEVICE)
                y = sample['y'].to(DEVICE)
                out = model(x, delay)
                val_loss += F.mse_loss(out, y).item()
                preds.append(out.cpu().numpy())
                targets.append(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        preds, targets = np.concatenate(preds), np.concatenate(targets)

        print(f"Epoch {epoch} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
        if input_dim > 1:
            print(f"  → Val MAE: {mean_absolute_error(targets, preds):.4f} | R²: {r2_score(targets, preds):.4f}")

    torch.cuda.empty_cache()
    # ==== SAVE MODEL ====
    save_path = os.path.join(SAVE_PATH, f"model_{key}.pt")
    torch.save(model.state_dict(), save_path)

    # ==== PLOT LOSSES ====
    plt.figure()
    plt.plot(loss_history, label="Train")
    plt.plot(val_loss_history, label="Val")
    plt.title(f"Loss Curve - {key}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"loss_curve_{key}.png"))

    # ==== PLOT PRED VS TRUE ====
    plt.figure(figsize=(6, 4))
    for i in range(min(PLOT_SAMPLES, len(preds))):
        plt.plot(targets[i], label=f"GT_{i}", linestyle='--')
        plt.plot(preds[i], label=f"Pred_{i}")
    plt.title(f"Pred vs GT - {key}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, f"compare_curve_{key}.png"))

# ========== TEST EVALUATION ==========
def evaluate_on_testset(model, key, input_dim):
    test_set = V2XDataset("/data/jqy/V2X-Sim-2.0/V2X-Sim-2-det/test/agent1", key)
    test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=False)

    model.eval()
    preds, targets = [], []
    test_loss = 0
    with torch.no_grad():
        for sample in test_loader:
            x = sample['x'].to(DEVICE)
            delay = sample['delay'].to(DEVICE)
            y = sample['y'].to(DEVICE)
            out = model(x, delay)
            test_loss += F.mse_loss(out, y).item()
            if key == 'num_sensor':
                pred_class = out.argmax(dim=1).cpu().numpy() + 3  # 回到原始标签
                true_class = y.view(-1).long().cpu().numpy()
                preds.append(pred_class)
                targets.append(true_class)
            else:
                preds.append(out.cpu().numpy())
                targets.append(y.cpu().numpy())


    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    test_loss /= len(test_loader)
    if key == 'num_sensor':
        acc = (np.array(preds) == np.array(targets)).mean()
        print(f"[TEST] {key}: Accuracy = {acc:.4f}")
    else:
        print(f"[TEST] {key}: MSE={test_loss:.4f} | MAE={mean_absolute_error(targets, preds):.4f} | R²={r2_score(targets, preds):.4f}")

    if input_dim > 1:
        for i in range(input_dim):
            mae_i = mean_absolute_error(targets[:, i], preds[:, i])
            r2_i = r2_score(targets[:, i], preds[:, i])
            print(f" → [dim {i}] MAE={mae_i:.4f} | R²={r2_i:.4f}")
            plt.figure()
            plt.plot(targets[:PLOT_SAMPLES, i].T, 'k--', alpha=0.5, label='GT')
            plt.plot(preds[:PLOT_SAMPLES, i].T, 'r-', alpha=0.5, label='Pred')
            plt.title(f"Test Prediction - {key} [dim {i}]")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(SAVE_PATH, f"test_pred_curve_{key}_dim{i}.png")
            plt.savefig(fig_path)
        
# ========== MAIN ==========
if __name__ == "__main__":
    for key, dim in FIELDS.items():
        train_field(key, dim)
        print("Running test evaluation...")
        model = TransformerPredictor(input_dim=dim, num_classes=4 if key == 'num_sensor' else None).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH, f"model_{key}.pt")))
        evaluate_on_testset(model, key, dim)
