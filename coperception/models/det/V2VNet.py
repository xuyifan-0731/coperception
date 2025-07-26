import torch
import coperception.utils.convolutional_rnn as convrnn
from coperception.models.det.base import IntermediateModelBase
import torch.nn.functional as F


class V2VNet(IntermediateModelBase):
    """V2V Net

    https://arxiv.org/abs/2008.07519

    """

    def __init__(
        self,
        config,
        gnn_iter_times,
        layer,
        layer_channel,
        in_channels=13,
        num_agent=5,
        compress_level=0,
        only_v2i=False,
    ):
        super().__init__(
            config,
            layer,
            in_channels,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

        self.layer_channel = layer_channel
        self.gnn_iter_num = gnn_iter_times
        self.convgru = convrnn.Conv2dGRU(
            in_channels=self.layer_channel * 2,
            out_channels=self.layer_channel,
            kernel_size=3,
            num_layers=1,
            bidirectional=False,
            dilation=1,
            stride=1,
        )
        self.compress_level = compress_level

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
                print(f"⚠️ Agent {i} has empty feature (batch size 0), replacing with zeros.")
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

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size):
        #print("🔥 forward 被调用")
        #print("num_agent_tensor shape:", num_agent_tensor.shape)
        #print("bevs shape:", bevs.shape)
        #print("batch_size 参数:", batch_size)
        # trans_matrices [batch 5 5 4 4]
        # num_agent_tensor, shape: [batch, num_agent]; how many non-empty agent in this scene
        #print("forward-num_agent_tensor_length:", num_agent_tensor.shape)

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        encoded_layers = self.u_encoder(bevs)
        device = bevs.device

        feat_maps, size = super().get_feature_maps_and_size(encoded_layers)
        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_list = super().build_feature_list(batch_size, feat_maps)
        #修改
        # 检查
        #for idx, f in enumerate(feat_list):
        #    if f is None or not isinstance(f, torch.Tensor) or f.shape[0] == 0:
        #        print(f"🚨 feat_list[{idx}] 无效：{type(f)}, shape: {getattr(f, 'shape', 'None')}")
        #
        
        #修改
        #feat_list = V2VNet.pad_features_to_match(feat_list)
        #
        local_com_mat = super().build_local_communication_matrix(feat_list)  # [2 5 512 16 16] [batch, agent, channel, height, width]
        #修改
        #if local_com_mat is None:
        #    raise RuntimeError("build_local_communication_matrix 返回 None，请检查输入特征列表")
        #if local_com_mat is None:
        #    print("⚠️ 警告：通信矩阵为 None，使用全零替代")
         #   shape = (batch_size, self.agent_num, self.layer_channel, *size)
        #    local_com_mat = torch.zeros(shape, dtype=torch.float32, device=device)
        #
        local_com_mat_update = super().build_local_communication_matrix(feat_list)  # to avoid the inplace operation
        real_batch_size = num_agent_tensor.shape[0] 
        for b in range(real_batch_size):
            #print("b")
            #print(b)
            #print("num_agent_tensor_length")
            #print(len(num_agent_tensor))
            num_agent = num_agent_tensor[b, 0]
            
            agent_feat_list = list()
            for nb in range(self.agent_num):
                agent_feat_list.append(local_com_mat[b, nb])
                
            for _ in range(self.gnn_iter_num):

                updated_feats_list = []

                for i in range(num_agent):
                    self.neighbor_feat_list = []
                    all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]

                    if super().outage():
                        updated_feats_list.append(agent_feat_list[i])

                    else:
                        super().build_neighbors_feature_list(
                            b,
                            i,
                            all_warp,
                            num_agent,
                            local_com_mat,
                            device,
                            size,
                            trans_matrices,
                        )

                        mean_feat = torch.mean(
                            torch.stack(self.neighbor_feat_list), dim=0
                        )  # [c, h, w]
                        cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                        cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, c, h, w]
                        updated_feat, _ = self.convgru(cat_feat, None)
                        updated_feat = torch.squeeze(
                            torch.squeeze(updated_feat, 0), 0
                        )  # [c, h, w]
                        updated_feats_list.append(updated_feat)

                agent_feat_list = updated_feats_list

            for k in range(num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]
        
        feat_maps = super().agents_to_batch(local_com_mat_update)

        decoded_layers = super().get_decoded_layers(
            encoded_layers, feat_maps, batch_size
        )
        x = decoded_layers[0]

        cls_pred, loc_preds, result = super().get_cls_loc_result(x)
        return result
