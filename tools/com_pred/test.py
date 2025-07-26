import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import TransformerPredictor
from sklearn.preprocessing import MinMaxScaler
import joblib

# ========== CONFIG ==========
DATA_ROOT = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-det/test/agent1"
MODEL_BASE_PATH = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-com-pred-model/agent1"
SAVE_DIR = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-com-pred-result/test/agent1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== FEATURE EXTRACTION ==========
def extract_feature(data, key, scaler=None):
    try:
        val = data[key]
        if key == 'reg_target_sparse':
            return val.mean(axis=0).squeeze()
        elif key in ['voxel_indices_0', 'voxel_indices_teacher', 'voxel_indices_teacher_no_cross_road']:
            return val.mean(axis=0)
        elif key == 'gt_max_iou':
            return val[:, :2].mean(axis=0)
        elif key in ['label_sparse', 'allocation_mask', 'reg_loss_mask']:
            return np.array([val.mean()])
        elif key in ['vis_occupy_indices', 'vis_free_indices']:
            return val[:3, :].mean(axis=1)
        elif key in ['target_agent_id', 'num_sensor']:
            return np.array([val])
        elif key in ['trans_matrices', 'trans_matrices_no_cross_road']:
            val = val.reshape(-1, 16)
            return val.mean(axis=0)
        feat = np.zeros(FIELDS[key])  # default fallback
        if scaler:
            feat = scaler.transform(feat.reshape(-1, 1)).flatten()
        return feat
    except:
        return np.zeros(FIELDS[key])

# ========== SCENE UTILS ==========
def get_test_scenes():
    folders = os.listdir(DATA_ROOT)
    return sorted(set(int(f.split('_')[0]) for f in folders if '_' in f))

def load_input(scene, frames, key):
    feats = []
    for f in frames:
        path = os.path.join(DATA_ROOT, f"{scene}_{f}", "0.npy")
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True).item()
        feats.append(extract_feature(data, key,scalar))
        if len(feats) < 5:
            feats += [np.zeros(FIELDS[key])] * (5 - len(feats))
        elif len(feats) > 5:
            feats = feats[:5]
    return np.stack(feats) if feats else np.zeros((5, FIELDS[key]))

def load_target(scene, ref_frame, delay, key):
    target_f = min(ref_frame + delay, 99)
    path = os.path.join(DATA_ROOT, f"{scene}_{target_f}", "0.npy")
    if not os.path.exists(path):
        return np.zeros(FIELDS[key])
    data = np.load(path, allow_pickle=True).item()
    return extract_feature(data, key ,scalar)

# ========== EVAL ==========
def evaluate_all():
    scenes = get_test_scenes()
    report = {}

    for key, dim in FIELDS.items():
        model_path = os.path.join(MODEL_BASE_PATH, f"model_{key}.pt")
        scaler_path = os.path.join(MODEL_BASE_PATH, f"scaler_{key}.pkl")
        if not os.path.exists(model_path):
            print(f"[Skip] No model for {key}")
            continue
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        print(f"\n=== Evaluating {key} ===")
        num_classes = 4 if key == 'num_sensor' else None
        model = TransformerPredictor(input_dim=dim, num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        preds, gts = [], []
        for scene in scenes:
            for ref in range(95):
                frames = list(range(ref, ref + 3))
                delay = 2
                x = load_input(scene, frames, key)
                xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([[delay]], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    pred = model(xt, dt).cpu().numpy().squeeze()
                gt = load_target(scene, frames[-1], delay, key)
                preds.append(pred)
                gts.append(gt)

        preds, gts = np.array(preds), np.array(gts)
        mse = ((preds - gts) ** 2).mean(axis=0)
        mse_total = mse.mean()
        report[key] = (mse.tolist(), float(mse_total))

        # plot
        plt.figure(figsize=(10, 4))
        if dim == 1:
            plt.plot(preds, '--', label="Pred")
            plt.plot(gts, label="GT")
        else:
            for i in range(dim):
                plt.plot(preds[:, i], '--', label=f"Pred {i}")
                plt.plot(gts[:, i], label=f"GT {i}")

        plt.title(f"{key} | MSE: {mse_total:.4f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{key}_compare.png"))
        plt.close()

    # write report
    with open(os.path.join(SAVE_DIR, "summary_report.txt"), "w") as f:
        for k, (mse_arr, mse_mean) in report.items():
            f.write(f"{k}: MSE per dim = {mse_arr}, Avg MSE = {mse_mean:.6f}\n")
    print("\nEvaluation summary saved to summary_report.txt")

if __name__ == '__main__':
    evaluate_all()
