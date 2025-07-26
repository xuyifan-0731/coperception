import numpy as np

def group_mean_voxel_indices(indices: np.ndarray) -> np.ndarray:
    """
    对形如 (N, 3) 的 voxel_indices 数组按第一列（label）分组，
    并对剩余列计算平均值后取整数，返回格式与输入类似的数组：
    每行 [label, mean_x_int, mean_y_int]。

    Parameters
    ----------
    indices : np.ndarray
        输入数组，shape (N, 3)，第一列为 label，后两列为特征。

    Returns
    -------
    np.ndarray
        输出数组，shape (M, 3)，M = 不同 label 数量。
        第 0 列是每个 label，第 1、2 列是对应特征均值四舍五入后的整数值。
    """
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] < 2:
        raise ValueError("indices must be a 2D array with at least 2 columns")

    labels = indices[:, 2].astype(int)
    features = indices[:, :2]

    unique_labels = np.unique(labels)

    # 将输出也设为整数类型
    result = np.zeros((unique_labels.size, indices.shape[1]), dtype=int)

    for i, lbl in enumerate(unique_labels):
        mask = (labels == lbl)
        mean_feats = features[mask].mean(axis=0)
        # 四舍五入并转 int
        mean_int = np.round(mean_feats).astype(int)

        result[i, 0]    = lbl
        result[i, 1:]   = mean_int

    return result

def sparse_to_dense(indices: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """
    Convert sparse voxel index list to a dense occupancy grid.

    Args:
        indices (np.ndarray): Array of shape [N, 3], dtype=int, each row is [z, y, x].
        grid_shape (tuple): The shape (Z, Y, X) of the desired dense grid.

    Returns:
        np.ndarray: Dense occupancy grid of shape grid_shape, dtype=uint8,
                    with 1 at occupied voxels and 0 elsewhere.
    """
    # Initialize a zero grid
    grid = np.zeros(grid_shape, dtype=np.uint8)

    # Unpack indices
    z, y, x = indices[:, 0], indices[:, 1], indices[:, 2]

    # Clip to valid range
    valid_mask = (
            (z >= 0) & (z < grid_shape[0]) &
            (y >= 0) & (y < grid_shape[1]) &
            (x >= 0) & (x < grid_shape[2])
    )
    z, y, x = z[valid_mask], y[valid_mask], x[valid_mask]

    # Mark occupied voxels
    grid[z, y, x] = 1

    return grid
import matplotlib.pyplot as plt

def visualize_voxel_indices(indices: np.ndarray):
    """
    Visualize sparse voxel indices in BEV (top-down) and 3D scatter.

    Args:
        indices (np.ndarray): Array of shape [N, 3], each row [z, y, x].
    """
    # Ensure indices is numpy array
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must be of shape [N, 3]")

    # Infer grid shape automatically
    z_max, y_max, x_max = indices.max(axis=0) + 1

    # 1. BEV occupancy map
    bev_shape = (int(z_max), int(y_max))
    print("BEV shape:", bev_shape)
    bev = np.zeros(bev_shape, dtype=np.uint8)
    y, x = indices[:, 0].astype(int), indices[:, 1].astype(int)
    valid = (y >= 0) & (y < bev_shape[0]) & (x >= 0) & (x < bev_shape[1])
    bev[y[valid], x[valid]] = 1

    plt.figure()
    plt.title("BEV Occupancy Map")
    plt.xlabel("X")
    plt.ylabel("Y (or Z)")
    plt.imshow(bev, origin='lower', aspect='equal')
    plt.tight_layout()
    plt.show()
    '''
    # 2. 沿 X 轴压缩投影（对每一行求和）
    #    这里 axis=1 是对 X 方向求和，得到长度为 bev_shape[0] 的向量
    proj_x = bev.sum(axis=1)

    plt.figure()
    plt.title("Projection Along X-axis")
    plt.xlabel("Row index (Y or Z)")
    plt.ylabel("Occupied count over X")
    plt.plot(np.arange(bev_shape[0]), proj_x, linewidth=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. 3D scatter plot
    z, y, x = indices[:, 0], indices[:, 1], indices[:, 2]
    valid = (z >= 0) & (z < z_max) & (y >= 0) & (y < y_max) & (x >= 0) & (x < x_max)
    z, y, x = z[valid], y[valid], x[valid]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)
    ax.set_title("3D Voxel Occupancy Scatter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()'''

for i in range(0,50):
    file = f"/Users/xuyifan/Downloads/V2X-Sim-det/train/agent1/15_{i}/0.npy"

    # 加载时用 allow_pickle=True，这样才能 load 出 object 数组
    feat = np.load(file, allow_pickle=True)

    # 先看看 feat 的形状和 dtype
    print(feat.shape, feat.dtype)
    # 比如输出可能是 () object —— 意味着这是一个 0-d 的 object ndarray

    # 如果是 0-d 的 object ndarray，可以用 item() 拿到它里面的 dict
    data_dict = feat.item()

    # 现在就可以这样访问
    voxel_indices_0 = data_dict["voxel_indices_0"]

    visualize_voxel_indices(voxel_indices_0)



