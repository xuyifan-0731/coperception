import os
import numpy as np
import torch
from train import SimplePredictor, FIELDS

# ===== CONFIG =====
TEST_ROOT = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-det/test/agent1"
MODEL_ROOT = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-com-pred-model/agent1"
OUTPUT_ROOT = "/data/jqy/V2X-Sim-2.0/V2X-Sim-2-com-pred-result/npy/agent1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

FIELDS = {
    'reg_target_sparse': 6,
    #'voxel_indices_0': 3,
    #'gt_max_iou': 2,
    #'label_sparse': 1,
    #'allocation_mask': 1,
    #'reg_loss_mask': 1,
    #'vis_occupy_indices': 3,
    #'vis_free_indices': 3,
    #'target_agent_id': 1,
    #'num_sensor': 1,
    #'trans_matrices': 16,
    #'trans_matrices_no_cross_road': 16,
    #'voxel_indices_teacher': 3,
    #'voxel_indices_teacher_no_cross_road': 3
}
# ===== Helper: load test scene/frame =====
def get_all_scene_frame_pairs(test_root):
    pairs = []
    for name in os.listdir(test_root):
        if not name.endswith("_0.npy"):
            continue
        try:
            scene, frame = map(int, name.replace(".npy", "").split("_"))
            pairs.append((scene, frame))
        except:
            continue
    return sorted(set(pairs))

# ===== Predict single scene-frame =====
def load_input_feature(scene, frame, key, input_dim):
    x_list = []
    for offset in [-4, -3, -2, -1, 0]:  # last 5 frames
        idx = max(0, frame + offset)
        npy_path = os.path.join(TEST_ROOT, f"{scene}_{idx}", "0.npy")
        if not os.path.exists(npy_path):
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        val = data.get(key)
        if val is None:
            continue
        # mimic extract_feature
        if key == 'reg_target_sparse':
            feat = val.mean(axis=0).squeeze()
        elif key in ['voxel_indices_0', 'voxel_indices_teacher', 'voxel_indices_teacher_no_cross_road']:
            feat = val.mean(axis=0)
        elif key == 'gt_max_iou':
            feat = val[:, :2].mean(axis=0)
        elif key in ['label_sparse', 'allocation_mask', 'reg_loss_mask']:
            feat = np.array([val.mean()])
        elif key in ['vis_occupy_indices', 'vis_free_indices']:
            feat = val[:3, :].mean(axis=1)
        elif key in ['target_agent_id', 'num_sensor']:
            feat = np.array([val])
        elif key in ['trans_matrices', 'trans_matrices_no_cross_road']:
            feat = val.reshape(-1)
        else:
            feat = np.zeros(input_dim)
        x_list.append(feat)

    # pad to 5
    if len(x_list) < 5:
        x_list += [np.zeros(input_dim)] * (5 - len(x_list))
    return np.stack(x_list)

# ===== Run prediction and save .npy =====
import matplotlib.pyplot as plt

def generate_all_predictions():
    for scene in range(100):
        for frame in range(95):  # 留出延迟空间
            pred_dict = {}
            for key, dim in FIELDS.items():
                model_path = os.path.join(MODEL_ROOT, f"model_{key}.pt")
                print(model_path)
                if not os.path.exists(model_path):
                    print(f"[Skip] no model for {key}")
                    continue

                model = SimplePredictor(input_dim=dim).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()

                x_feat = load_input_feature(scene, frame, key, dim)
                x_tensor = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                delay_tensor = torch.tensor([[2]], dtype=torch.long).to(DEVICE)

                with torch.no_grad():
                    pred = model(x_tensor, delay_tensor).cpu().numpy().squeeze()
                    pred_dict[key] = pred

            gt_path = os.path.join(TEST_ROOT, f"{scene}_{frame}", "0.npy")
            if os.path.exists(gt_path):
                gt_data = np.load(gt_path, allow_pickle=True).item()
                mse_report = {}
                for key in pred_dict:
                    if key in gt_data:
                        try:
                            gt_feat = load_input_feature(scene, frame, key, FIELDS[key])[-1]  # use current frame only
                            pred_feat = pred_dict[key]
                            mse = ((pred_feat - gt_feat) ** 2).mean()
                            mse_report[key] = mse

                            # Optional: plot prediction vs GT (first dim)
                            plt.figure()
                            plt.plot(pred_feat, label="Pred", linestyle="--")
                            plt.plot(gt_feat, label="GT")
                            plt.title(f"Scene {scene} Frame {frame} - {key} MSE={mse:.4f}")
                            plt.legend()
                            plt.tight_layout()
                            plot_path = os.path.join(OUTPUT_ROOT, f"plot_{scene}_{frame}_{key}.png")
                            plt.savefig(plot_path)
                            plt.close()
                        except Exception as e:
                            print(f"[Warn] Could not compare {key}: {e}")

                # save comparison report
                report_path = os.path.join(OUTPUT_ROOT, f"{scene}_{frame}_mse.txt")
                with open(report_path, "w") as f:
                    for k, v in mse_report.items():
                        f.write(f"{k}: {v:.6f}\n")

            save_path = os.path.join(OUTPUT_ROOT, f"{scene}_{frame}.npy")
            np.save(save_path, pred_dict)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    generate_all_predictions()
