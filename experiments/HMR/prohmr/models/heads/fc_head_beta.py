import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from yacs.config import CfgNode

class FCHeadBeta(nn.Module):

    def __init__(self, cfg: CfgNode, fc_head_inchannel=None, deep=False, condition_on_pose=False):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(FCHeadBeta, self).__init__()
        self.cfg = cfg
        self.condition_on_pose = condition_on_pose
        if self.condition_on_pose:
            in_channel = fc_head_inchannel + 144
        else:
            in_channel = fc_head_inchannel
        if not deep:
            self.layers = nn.Sequential(nn.Linear(in_channel,
                                                  cfg.MODEL.FC_HEAD.NUM_FEATURES),
                                                  nn.ReLU(inplace=False),
                                                  nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, 10))
            nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)
        else:
            self.layers = nn.Sequential(nn.Linear(in_channel, 1024),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 10))

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        # init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[None, None]
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None, None]

        # self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_betas', init_betas)

    def forward(self, smpl_params, pred_pose, feats) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass.
        Args:
            smpl_params (Dict): Dictionary containing predicted SMPL parameters.
            feats (torch.Tensor): Tensor of shape (N, C) containing the features computed by the backbone.
        Returns:
            pred_betas (torch.Tensor): Predicted SMPL betas.
            pred_cam (torch.Tensor): Predicted camera parameters.
        """
        # pred_pose: [bs, n_sample, 144]

        batch_size = feats.shape[0]
        num_samples = smpl_params['body_pose'].shape[1]

        if not self.condition_on_pose:
            offset = self.layers(feats)  # [bs, 10]
            offset = offset.reshape(batch_size, 1, 10).repeat(1, num_samples, 1)  # [bs, n_sample, 10]
        else:
            feats = feats.unsqueeze(1).repeat(1, num_samples, 1)  # [bs, n_sample, feat_dim]
            feats = torch.cat([feats, pred_pose], dim=-1)  # [bs, n_sample, feat_dim+144]
            offset = self.layers(feats.reshape(batch_size*num_samples, -1)).reshape(batch_size, num_samples, 10)

        pred_betas = offset + self.init_betas
        return pred_betas
