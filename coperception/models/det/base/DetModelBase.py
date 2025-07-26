from coperception.models.det.backbone.Backbone import *
import numpy as np


class DetModelBase(nn.Module):
    """Abstract class. The super class for all detection models.

    Attributes:
        motion_state (bool): To return motion state in the loss calculation method or not
        out_seq_len (int): Length of output sequence
        box_code_size (int): The specification for bounding box encoding.
        category_num (int): Number of categories.
        use_map (bool): use_map
        anchor_num_per_loc (int): Anchor number per location.
        classification (nn.Module): The classification head.
        regression (nn.Module): The regression head.
        agent_num (int): The number of agent (including RSU and vehicles)
        kd_flag (bool): Required for DiscoNet.
        layer (int): Collaborate at which layer.
        p_com_outage (float): The probability of communication outage.
        neighbor_feat_list (list): The list of neighbor features.
        tg_agent (tensor): Features of the current target agent.
    """

    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        p_com_outage=0.0,
        num_agent=5,
        only_v2i=False
    ):
        super(DetModelBase, self).__init__()

        self.motion_state = config.motion_state
        self.out_seq_len = 1 if config.only_det else config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.agent_num = num_agent
        self.kd_flag = kd_flag
        self.layer = layer
        self.p_com_outage = p_com_outage
        self.neighbor_feat_list = []
        self.tg_agent = None
        self.only_v2i = only_v2i

    def agents_to_batch(self, feats):
        """Concatenate the features of all agents back into a bacth.

        Args:
            feats (tensor): features

        Returns:
            The concatenated feature matrix of all agents.
        """
        feat_list = []
        for i in range(self.agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)

        feat_mat = torch.flip(feat_mat, (2,))

        return feat_mat

    def get_feature_maps_and_size(self, encoded_layers: list):
        """Get the features of the collaboration layer and return the corresponding size.

        Args:
            encoded_layers (list): The output from the encoder.

        Returns:
            Feature map of the collaboration layer and the corresponding size.
        """
        feature_maps = encoded_layers[self.layer]

        size_tuple = (
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 128, 64, 64),
            (1, 256, 32, 32),
            (1, 512, 16, 16),
        )
        size = size_tuple[self.layer]

        feature_maps = torch.flip(feature_maps, (2,))
        return feature_maps, size

    #修改
    
    @staticmethod
    def pad_features_to_match(feat_list):
        """
        Ensure all tensors in the feature list have the same shape (including batch size).
        If a tensor is empty or has smaller batch size, pad with zeros.
        """
        # 先找出目标 shape：最大 batch size 和每个维度的最大值
        max_shape = list(feat_list[0].shape)
        for feat in feat_list[1:]:
            for i in range(len(max_shape)):  # 包括 dim=0 (batch)
                max_shape[i] = max(max_shape[i], feat.shape[i])

        padded_feats = []
        for i, feat in enumerate(feat_list):
            # 处理 batch size 为 0
            if feat.shape[0] == 0:
                #print(f"⚠️ Agent {i} has empty feature (batch size 0), replacing with zeros.")
                feat = torch.zeros(max_shape, dtype=feat_list[0].dtype, device=feat_list[0].device)
                padded_feats.append(feat)
                continue

            pad_sizes = []
            for d in reversed(range(len(max_shape))):  # reverse for F.pad
                diff = max_shape[d] - feat.shape[d]
                pad_sizes.extend([0, diff])

            feat = F.pad(feat, pad_sizes, mode='constant', value=0)
            padded_feats.append(feat)

        return padded_feats


        #

    def build_feature_list(self, batch_size: int, feat_maps) -> list:
        """Get the feature maps for each agent

        e.g: [10 512 16 16] -> [2 5 512 16 16] [batch size, agent num, channel, height, width]

        Args:
            batch_size (int): The batch size.
            feat_maps (tensor): The feature maps of the collaboration layer.

        Returns:
            A list of feature maps for each agent.
        """
        feature_map = {}
        feature_list = []

        for i in range(self.agent_num):
            feature_map[i] = torch.unsqueeze(
                feat_maps[batch_size * i : batch_size * (i + 1)], 1
            )
            feature_list.append(feature_map[i])

        return feature_list

    @staticmethod
    def build_local_communication_matrix(feature_list: list):
        """Concatendate the feature list into a tensor.

        Args:
            feature_list (list): The input feature list for each agent.

        Returns:
            A tensor of concatenated features.
        """
        # #存在一些agent的特征维度不对齐的问题
        # clean_feature_list = []

        # for idx, feat in enumerate(feature_list):
        #     # 1. 检查是否为 None 或非法类型
        #     if feat is None or not isinstance(feat, torch.Tensor):
        #         print(f"🚫 Agent {idx} 的特征无效（None 或非 tensor），使用 zeros 替代")
        #         feat = torch.zeros(
        #             (1, 1, 256, 32, 32), dtype=torch.float32, device=feature_list[0].device
        #         )

        #     # 2. 检查 shape 是否为空
        #     elif feat.shape[0] == 0:
        #         print(f"⚠️ Warning: Empty feature for agent {idx}, using zeros instead")
        #         feat = torch.zeros(
        #             (1, 1, 256, 32, 32), dtype=torch.float32, device=feature_list[0].device
        #         )

        #     # 3. 加入列表
        #     clean_feature_list.append(feat)

        # # 4. 检查维度一致性
        # shapes = [f.shape for f in clean_feature_list]
        # if not all(s == shapes[0] for s in shapes):
        #     print(f"❌ Feature shape mismatch detected:")
        #     for i, s in enumerate(shapes):
        #         print(f"  Agent {i}: shape = {s}")
        #     raise RuntimeError("All features must have the same shape for torch.cat")

        # # 5. 拼接特征：在维度 1（agent）上拼接 这个是原代码
        # #
        #return torch.cat(tuple(clean_feature_list), dim=1)
        #for idx, feat in enumerate(feature_list):
            #print(f"Feature {idx} shape before padding: {feat.shape}")
        #feature_list = DetModelBase.pad_features_to_match(feature_list)
        #for idx, feat in enumerate(feature_list):
            #print(f"Feature {idx} shape after padding: {feat.shape}")

        return torch.cat(tuple(feature_list), 1)

    def outage(self) -> bool:
        """Simulate communication outage according to self.p_com_outage.

        Returns:
            A bool indicating if the communication outage happens.
        """
        return np.random.choice(
            [True, False], p=[self.p_com_outage, 1 - self.p_com_outage]
        )

    @staticmethod
    def feature_transformation(
        b, j, agent_idx, local_com_mat, all_warp, device, size, trans_matrices
    ):
        """Transform the features of the other agent (j) to the coordinate system of the current agent.

        Args:
            b (int): The index of the sample in current batch.
            j (int): The index of the other agent.
            local_com_mat (tensor): The local communication matrix. Features of all the agents.
            all_warp (tensor): The warp matrix for current sample for the current agent.
            device: The device used for PyTorch.
            size (tuple): Size of the feature map.

        Returns:
            A tensor of transformed features of agent j.
        """
        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)

        tfm_ji = trans_matrices[b, j, agent_idx]
        M = (
            torch.hstack((tfm_ji[:2, :2], -tfm_ji[:2, 3:4])).float().unsqueeze(0)
        )  # [1,2,3]

        mask = torch.tensor([[[1, 1, 4 / 128], [1, 1, 4 / 128]]], device=M.device)

        M *= mask

        grid = F.affine_grid(M, size=torch.Size(size))
        warp_feat = F.grid_sample(nb_agent, grid).squeeze()
        return warp_feat

    def build_neighbors_feature_list(
        self,
        b,
        agent_idx,
        all_warp,
        num_agent,
        local_com_mat,
        device,
        size,
        trans_matrices,
    ) -> None:
        """Append the features of the neighbors of current agent to the neighbor_feat_list list.

        Args:
            b (int): The index of the sample in current batch.
            agent_idx (int): The index of the current agent.
            all_warp (tensor): The warp matrix for current sample for the current agent.
            num_agent (int): The number of agents.
            local_com_mat (tensor): The local communication matrix. Features of all the agents.
            device: The device used for PyTorch.
            size (tuple): Size of the feature map.
        """
        for j in range(num_agent):
            if j != agent_idx:
                if self.only_v2i and agent_idx != 0 and j != 0:
                    continue
                
                warp_feat = DetModelBase.feature_transformation(
                    b,
                    j,
                    agent_idx,
                    local_com_mat,
                    all_warp,
                    device,
                    size,
                    trans_matrices,
                )

                self.neighbor_feat_list.append(warp_feat)

    def get_decoded_layers(self, encoded_layers, feature_fuse_matrix, batch_size):
        """Replace the collaboration layer of the output from the encoder with fused feature maps.

        Args:
            encoded_layers (list): The output from the encoder.
            feature_fuse_matrix (tensor): The fused feature maps.
            batch_size (int): The batch size.

        Returns:
            A list. Output from the decoder.
        """
        encoded_layers[self.layer] = feature_fuse_matrix
        decoded_layers = self.decoder(*encoded_layers, batch_size, kd_flag=self.kd_flag)
        return decoded_layers

    def get_cls_loc_result(self, x):
        """Get the classification and localization result.

        Args:
            x (tensor): The output from the last layer of the decoder.

        Returns:
            cls_preds (tensor): Predictions of the classification head.
            loc_preds (tensor): Predications of the localization head.
            result (dict): A dictionary of classificaion, localization, and optional motion state classification result.
        """
        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(
            -1,
            loc_preds.size(1),
            loc_preds.size(2),
            self.anchor_num_per_loc,
            self.out_seq_len,
            self.box_code_size,
        )

        # loc_pred (N * T * W * H * loc)
        result = {"loc": loc_preds, "cls": cls_preds}

        # MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0], -1, motion_cat)
            result["state"] = motion_cls_preds

        return cls_preds, loc_preds, result


class ClassificationHead(nn.Module):
    """The classificaion head."""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()

        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channel,
            category_num * anchor_num_per_loc,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class SingleRegressionHead(nn.Module):
    """The regression head."""

    def __init__(self, config):
        super(SingleRegressionHead, self).__init__()

        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        out_seq_len = 1 if config.only_det else config.pred_len

        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(
                        channel,
                        anchor_num_per_loc * box_code_size * out_seq_len,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
            else:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(
                        128,
                        anchor_num_per_loc * box_code_size * out_seq_len,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )

    def forward(self, x):
        box = self.box_prediction(x)

        return box
