import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.utils import AverageMeter
from coperception.utils.data_util import apply_pose_noise
from coperception.utils.mean_ap import eval_map
import copy

import glob
import os
import wandb


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def _parse_seq_idx(filename: str):
    """从 '.../agent_xxx_seqIdx/...' ⽂件名中提取 seq_name 与帧号 idx"""
    cut = filename[filename.rfind("agent") + 7:]
    seq_name = cut[: cut.rfind("_")]
    idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
    return seq_name, idx


@torch.no_grad()
def validate(
    faf_module,
    val_data_loader,
    device,
    config,
    num_agent,
    args,                       # 仍按你的习惯传 argparse 对象进来
    flag="v2v",              # "upperbound" | "lowerbound_box_com" | "disco"...
    pose_noise=0.0,             # 位姿噪声⼤⼩，0 表示不加
    apply_late_fusion=0,        # 0 关，1 开
    save_fig_path=None,         # list[str]  ⽤于保存可视化截图
    tracking_path=None,         # list[str]  ⽤于 MOT det 输出
    logger_root="logs",         # 如果想单独指定日志⽬录
):
    """验证阶段：计算 loss + 各 agent 局部 mAP + 平均 mAP，并输出可视化/跟踪结果"""
    faf_module.model.eval()

    # ---------① 统计量初始化---------
    total_loss = total_cls_loss = total_loc_loss = 0.0
    det_results_local  = [[] for _ in range(num_agent)]
    annotations_local  = [[] for _ in range(num_agent)]
    tracking_file      = [set() for _ in range(num_agent)]   # 记录已写过的 scene

    # ---------② 主循环---------
    for cnt, sample in enumerate(val_data_loader):
        t = time.time()
        if cnt >= 2:
            break

        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,                 # list[dict]  → cal_local_mAP ⽤
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        # 取 batch 第⼀条的⽂件名，后⾯拿来输出场景帧号
        filename0 = filenames[0]

        # ----------数据拼接(batch → (B*C,H,W,...) )----------
        trans_matrices   = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents   = torch.stack(tuple(num_agent_list), 1)

        # 给位姿加噪声（如需要）
        if pose_noise > 0:
            apply_pose_noise(pose_noise, trans_matrices)

        # RSU 占位的处理
        if not args.rsu:
            num_all_agents -= 1

        # upperbound ⽤ teacher points，其余⽤本地 points
        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target    = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map   = torch.cat(tuple(anchors_map_list), 0)
        vis_maps      = torch.cat(tuple(vis_maps_list), 0)
        padded_voxel_points = padded_voxel_points.amax(dim=-1, keepdim=True)

        # ---------③ 组装 forward 输入---------
        data = {
            "bev_seq":        padded_voxel_points.to(device),
            "labels":         label_one_hot.to(device),
            "reg_targets":    reg_target.to(device),
            "anchors":        anchors_map.to(device),
            "vis_maps":       vis_maps.to(device),
            "reg_loss_mask":  reg_loss_mask.to(device).type(torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent":        num_all_agents.to(device),
            "trans_matrices":   trans_matrices.to(device),
        }

        # ---------④ forward + loss / pred---------
        if flag == "lowerbound_box_com":
            loss, cls_loss, loc_loss, result = faf_module.predict_all_with_box_com(
                data, data["trans_matrices"]
            )
        elif flag == "disco":
            loss, cls_loss, loc_loss, result, save_agent_weight_list = \
                faf_module.predict_all(data, 1, num_agent=num_agent)
        else:
            loss, cls_loss, loc_loss, result = faf_module.predict_all(
                data, 1, num_agent=num_agent
            )

        total_loss     += loss
        total_cls_loss += cls_loss
        total_loc_loss += loc_loss

        # ---------⑤ 定义⽀持变量---------
        box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]
        eval_start_idx = 1 if args.rsu else 0   # 如果含 RSU，第 0 个 agent 是 RSU

        # ---------⑥ 局部 mAP / 可视化 / tracking---------
        for k in range(eval_start_idx, num_agent):

            # ---------⑥‑1 late‑fusion 备份---------
            box_colors = None
            if apply_late_fusion == 1 and len(result[k]) != 0:
                pred_restore      = result[k][0][0][0]["pred"]
                score_restore     = result[k][0][0][0]["score"]
                selected_idx_rest = result[k][0][0][0]["selected_idx"]

            # 只抽出 agent‑k 的单帧张量
            data_agents = {
                "bev_seq":     torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors":     torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
            }
            temp = gt_max_iou[k]
            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]

            # -- late fusion 上⾊
            if apply_late_fusion == 1 and len(result[k]) != 0:
                box_colors = late_fusion(
                    k, num_agent, result, trans_matrices, box_color_map
                )

            result_temp = result[k]

            temp = {
                "bev_seq":     data_agents["bev_seq"][0, -1].cpu().numpy(),
                "result":      [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou":  data_agents["gt_max_iou"],
            }
            # 计算并存储每帧的 det/gt（后⾯算 mAP）
            det_results_local[k], annotations_local[k] = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k]
            )

            # ----------可视化----------
            '''
            if args.visualization:
                filename = str(filename0[0][0])
                seq_name, idx = _parse_seq_idx(filename)  # ⾃定义⼀个拆⽂件名⼩⽅法
                seq_save = os.path.join(save_fig_path[k], seq_name)
                check_folder(seq_save)
                vis_path = os.path.join(seq_save, f"{idx}.png")
                visualization(
                    config, temp, box_colors, box_color_map,
                    apply_late_fusion, vis_path
                )'''

            # ----------tracking----------
            '''
            if args.tracking:
                scene, frame = filename0[0][0].split("/")[-2].split("_")
                det_file_path = os.path.join(tracking_path[k], f"det_{scene}.txt")
                if scene not in tracking_file[k]:
                    tracking_file[k].add(scene)
                    det_file = open(det_file_path, "w")
                else:
                    det_file = open(det_file_path, "a")

                det_corners = get_det_corners(config, copy.deepcopy(temp))
                for ic, c in enumerate(det_corners):
                    det_file.write(
                        ",".join(
                            [
                                str(int(frame) + 1), "-1",
                                f"{c[0]:.2f}", f"{c[1]:.2f}",
                                f"{c[2]:.2f}", f"{c[3]:.2f}",
                                str(result_temp[0][0][0]["score"][ic]),
                                "-1", "-1", "-1",
                            ]
                        ) + "\n"
                    )
                det_file.close()'''

            # ----------恢复 late‑fusion 前的 pred----------
            if apply_late_fusion == 1 and len(result[k]) != 0:
                result[k][0][0][0]["pred"]        = pred_restore
                result[k][0][0][0]["score"]       = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_rest

    # ---------⑦ 计算 mAP---------
    logger_root = os.path.join(
        logger_root, f"{flag}_eval", "with_rsu" if args.rsu else "no_rsu"
    )
    os.makedirs(logger_root, exist_ok=True)
    log_file_path = os.path.join(logger_root, "log_test.txt")
    log_file = open(log_file_path, "w")

    def _log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    # 逐 agent + 汇总
    mean_ap_local = []
    det_results_all_local, annotations_all_local = [], []
    for k in range(eval_start_idx, num_agent):
        if not det_results_local[k]:   # list 判空
            continue
        _log(f"Local mAP@0.5 from agent {k}")
        m05, _ = eval_map(det_results_local[k], annotations_local[k], iou_thr=0.5)
        mean_ap_local.append(m05)
        _log(f"Local mAP@0.7 from agent {k}")
        m07, _ = eval_map(det_results_local[k], annotations_local[k], iou_thr=0.7)
        mean_ap_local.append(m07)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    # 全 agent 平均
        try:
            m05_avg, _ = eval_map(det_results_all_local, annotations_all_local, iou_thr=0.5)
            m07_avg, _ = eval_map(det_results_all_local, annotations_all_local, iou_thr=0.7)
        except Exception as e:
            import traceback
            print(f"Error calculating mAP: {e}")
            m05_avg = 0
            m07_avg = 0
        mean_ap_local.extend([m05_avg, m07_avg])

    # ---------⑧ 打印最终结果---------
    print(f"mean_ap_local: {mean_ap_local}")
    _log(f"Quantitative evaluation results of model {args.resume}")
    for k in range(num_agent - (1 if args.rsu else 0)):
        _log(
            f"agent{k+1} mAP@0.5 = {mean_ap_local[k*2]:.4f}, "
            f"mAP@0.7 = {mean_ap_local[k*2+1]:.4f}"
        )
    _log(f"average local mAP@0.5 = {m05_avg:.4f}, average local mAP@0.7 = {m07_avg:.4f}")
    log_file.close()

    # ---------⑨ 返回供外部记录---------
    n_batches = len(val_data_loader)
    return {
        "val/loss":       total_loss / n_batches,
        "val/cls_loss":   total_cls_loss / n_batches,
        "val/loc_loss":   total_loc_loss / n_batches,
        "val/map@0.5":    m05_avg,
        "val/map@0.7":    m07_avg,
    }


def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch_size
    compress_level = args.compress_level
    auto_resume_path = args.auto_resume_path
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i
    shift = args.shift

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com" and args.warp_flag:
        flag = "when2com_warp"
    elif args.com in [
        "lowerbound",
        "v2v",
        "disco",
        "sum",
        "mean",
        "max",
        "cat",
        "agent",
        "when2com",
    ]:
        flag = args.com
    else:
        raise ValueError(f"com: {args.com} is not supported")

    config.flag = flag

    num_agent = args.num_agent
    # agent0 is the RSU
    # 数据加载
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    training_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=args.kd_flag,
        rsu=args.rsu,
        shift=shift,
    )
    training_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                      drop_last=True)

    # training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    print("Training dataset size:", len(training_dataset))

    val_dataset = V2XSimDet(
        dataset_roots=[f"{args.val_data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=args.kd_flag,
        rsu=args.rsu,
    )
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    print("Validation dataset size:", len(val_dataset))
    # 数据加载结束
    logger_root = args.logpath if args.logpath != "" else "logs"

    if not args.rsu:
        num_agent -= 1
    # 模型实例化
    if flag == "lowerbound" or flag == "upperbound":
        model = FaFNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif flag == "when2com" or flag == "when2com_warp":
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
            in_channels=1  # 这里改成1是因为把输入的高度压成1（实现在padded_voxel_point = padded_voxel_point.amax(dim=-1, keepdim=True))
        )
    elif flag == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    # model = nn.DataParallel(model)
    model = model.to(device)
    # 模型实例化结束
    # Adam 优化器、交叉熵 + 平滑 L1 损失（或 Focal Loss + 加权 L1）组合。
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    # 优化器结束
    # teacher模式
    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(
            model, teacher, config, optimizer, criterion, args.kd_flag
        )
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        faf_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print(
            "Load teacher model from {}, at epoch {}".format(
                args.resume_teacher, start_epoch_teacher
            )
        )
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)
        # print(modle,args.kd_flag)
    # teacher模式结束
    rsu_path = "with_rsu" if args.rsu else "no_rsu"
    model_save_path = check_folder(logger_root)
    model_save_path = check_folder(os.path.join(model_save_path, flag))

    if args.rsu:
        model_save_path = check_folder(os.path.join(model_save_path, "with_rsu"))
    else:
        model_save_path = check_folder(os.path.join(model_save_path, "no_rsu"))
        # print(model_save_path)
    # 恢复训练逻辑
    # 优先检查指定的 auto_resume_path 下是否有可用 checkpoint，否则使用 --resume 路径。
    # 加载模型、优化器、学习率调度器状态，并将 start_epoch 设为上次训练结束后的一轮 +1。
    # 如果 --resume 和 auto_resume_path 都为空，则从头开始训练，并新建日志文件。
    # check if there is valid check point file
    has_valid_pth = False
    for pth_file in os.listdir(os.path.join(auto_resume_path, f"{flag}/{rsu_path}")):
        if pth_file.startswith("epoch_") and pth_file.endswith(".pth"):
            has_valid_pth = True
            break

    if not has_valid_pth:
        print(
            f"No valid check point file in {auto_resume_path} dir, weights not loaded."
        )
        auto_resume_path = ""

    if args.resume == "" and auto_resume_path == "":
        log_file_name = os.path.join(model_save_path, "log.txt")
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
    else:
        if auto_resume_path != "":
            model_save_path = os.path.join(auto_resume_path, f"{flag}/{rsu_path}")
        else:
            model_save_path = args.resume[: args.resume.rfind("/")]

        # print(f"model save path: {model_save_path}")

        log_file_name = os.path.join(model_save_path, "log.txt")

        if os.path.exists(log_file_name):
            saver = open(log_file_name, "a")
        else:
            os.makedirs(model_save_path, exist_ok=True)
            saver = open(log_file_name, "w")

        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        if auto_resume_path != "":
            list_of_files = glob.glob(f"{model_save_path}/*.pth")
            latest_pth = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pth)
        else:
            checkpoint = torch.load(args.resume)

        start_epoch = checkpoint["epoch"] + 1
        faf_module.model.load_state_dict(checkpoint["model_state_dict"])
        faf_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        faf_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    # 恢复训练逻辑结束
    # 训练主循环
    for epoch in range(start_epoch, num_epochs + 1):
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter("Total loss", ":.6f")
        running_loss_class = AverageMeter(
            "classification Loss", ":.6f"
        )  # for cell classification error
        running_loss_loc = AverageMeter(
            "Localization Loss", ":.6f"
        )  # for state estimation error
        # 训练
        faf_module.model.train()
        # 1. 解包并拼接各 agent 数据
        t = tqdm(training_data_loader)
        for step, sample in enumerate(t):
            (
                padded_voxel_point_list,  # voxelized point cloud for individual agent
                padded_voxel_points_teacher_list,  # fused voxelized point cloud for all agents (multi-view)
                label_one_hot_list,  # one hot labels
                reg_target_list,  # regression targets
                reg_loss_mask_list,
                anchors_map_list,  # anchor boxes
                vis_maps_list,
                target_agent_id_list,
                num_agent_list,  # e.g. 6 agent in current scene: [6,6,6,6,6,6], 5 agent in current scene: [5,5,5,5,5,0]
                trans_matrices_list,
            # matrix for coordinate transformation. e.g. [batch_idx, j, i] ==> transformation matrix to transfer from agent i to j
            ) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)
            # 2. 可选添加位姿噪声 apply_pose_noise
            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)

            if not args.rsu:
                num_all_agents -= 1

            if flag == "upperbound":
                padded_voxel_point = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)
            # 3. 构造输入 dict data，包含 bev_seq、labels、anchors 等
            padded_voxel_point = padded_voxel_point.amax(dim=-1, keepdim=True)
            data = {
                "bev_seq": padded_voxel_point.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "vis_maps": vis_maps.to(device),
                "target_agent_ids": target_agent_id.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices": trans_matrices.to(device),
            }

            # 4. 若蒸馏：添加 bev_seq_teacher、kd_weight
            if args.kd_flag == 1:
                padded_voxel_points_teacher = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight
            # 5. 统计损失、打印进度
            loss, cls_loss, loc_loss = faf_module.step(
                data, batch_size, num_agent=num_agent
            )
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)

            if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
                print(f"Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}")
                sys.exit()

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(
                cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg
            )
            save_log = {"epoch": epoch,
                "train/loss": running_loss_disp.avg,
                "train/cls_loss": running_loss_class.avg,
                "train/loc_loss": running_loss_loc.avg,}
            if step % 2 == 0:
                validate_metrics = validate(faf_module, val_data_loader, device, config, num_agent, args)
                print(f"[Val] Epoch {epoch} | Step {step}|Loss: {validate_metrics['val/loss']:.4f} | mAP@0.5: {validate_metrics['val/m05_avg']:.4f}")
                save_log = {**validate_metrics,
                    **save_log,
                }
            wandb.log(save_log, step=step)

        faf_module.scheduler.step()
        # 6. 根据配置保存模型权重到指定目录
        # save model
        if need_log:
            saver.write(
                "{}\t{}\t{}\n".format(
                    running_loss_disp, running_loss_class, running_loss_loc
                )
            )
            saver.flush()
            if config.MGDA:
                save_dict = {
                    "epoch": epoch,
                    "encoder_state_dict": faf_module.encoder.state_dict(),
                    "optimizer_encoder_state_dict": faf_module.optimizer_encoder.state_dict(),
                    "scheduler_encoder_state_dict": faf_module.scheduler_encoder.state_dict(),
                    "head_state_dict": faf_module.head.state_dict(),
                    "optimizer_head_state_dict": faf_module.optimizer_head.state_dict(),
                    "scheduler_head_state_dict": faf_module.scheduler_head.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            else:
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": faf_module.model.state_dict(),
                    "optimizer_state_dict": faf_module.optimizer.state_dict(),
                    "scheduler_state_dict": faf_module.scheduler.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            torch.save(
                save_dict, os.path.join(model_save_path, "epoch_" + str(epoch) + ".pth")
            )
        # 每epoch评估一次val

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument(

        "--val_data",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV val data",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", default=0, type=int, help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", default=True, help="Visualize validation result"
    )
    parser.add_argument(
        "--com",
        default="",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--auto_resume_path",
        default="",
        type=str,
        help="The path to automatically reload the latest pth",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )
    parser.add_argument(
        "--shift",
        default=0,
        type=int,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    wandb.init(project="v2vdet", config=vars(args))
    print(args)
    main(args)
