import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from yacs.config import CfgNode
from prohmr.utils.geometry import rot6d_to_rotmat

class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class ShapeDec(nn.Module):
    def __init__(self, cfg: CfgNode, fc_head_inchannel=None):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(ShapeDec, self).__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(nn.Linear(fc_head_inchannel,
                                              cfg.MODEL.FC_HEAD.NUM_FEATURES),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, 10))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None]  # [1, 10]
        self.register_buffer('init_betas', init_betas)

    def forward(self, context):
        offset = self.layers(context)  # [bs, 10]
        pred_betas = offset + self.init_betas
        return pred_betas


class ShapeCamDec(nn.Module):
    def __init__(self, cfg: CfgNode, fc_head_inchannel=None):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(ShapeCamDec, self).__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(nn.Linear(fc_head_inchannel,
                                              cfg.MODEL.FC_HEAD.NUM_FEATURES),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[None]  # [1, 3]
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None]  # [1, 10]

        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_betas', init_betas)

    def forward(self, context):
        # context: [bs, context_dim]
        # offset = self.layers(feats).reshape(batch_size, 1, 13).repeat(1, num_samples, 1)
        offset = self.layers(context)  # [bs, 13]
        betas_offset = offset[:, :10]
        cam_offset = offset[:, 10:]
        pred_cam = cam_offset + self.init_cam
        pred_betas = betas_offset + self.init_betas
        return pred_betas, pred_cam



class BodyDec(nn.Module):
    def __init__(self, in_dim=24, context_dim=2048, hsize1=256, hsize2=512, out_type='params', pred_cam=True, cfg=None):
        super(BodyDec, self).__init__()
        self.out_type = out_type
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hsize1)
        self.fc2 = nn.Linear(in_features=hsize1, out_features=hsize2)
        self.res_block1 = ResnetBlockFC(size_in=hsize2+context_dim, size_out=hsize2, size_h=hsize2)
        self.res_block2 = ResnetBlockFC(size_in=hsize2, size_out=hsize2, size_h=hsize2)
        if out_type == 'params':
            out_dim = 144  # 6d rot for global_orient + pose
        elif out_type == 'markers':
            out_dim = 66 * 3
        elif out_type == 'joints':
            out_dim = 45 * 3
        elif out_type == 'joints+markers':
            out_dim = (45+66) * 3
        self.res_block3 = ResnetBlockFC(size_in=hsize2, size_out=out_dim, size_h=hsize2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # todo

        self.pred_cam = pred_cam
        if self.pred_cam:
            self.shape_cam_decoder = ShapeCamDec(cfg, context_dim)  # for beta, pred_cam
        else:
            self.shape_decoder = ShapeDec(cfg, context_dim)


    def forward(self, x, context):
        # input x: body bps encoding, [bs, n_sample, 45/66/xxx]
        # context: [bs, context_dim]
        batch_size = x.shape[0]
        num_samples = x.shape[1]
        x = self.relu(self.fc1(x))  # [bs, n_sample, hsize1]
        x = self.fc2(x)  # [bs, n_sample, hsize2]
        x = torch.cat([x, context.unsqueeze(1).repeat(1, num_samples, 1)], dim=-1)  # [bs, n_sample, hsize2+context_dim]
        x = self.res_block1(x)  # [bs, n_sample, hsize2]
        x = self.res_block2(x)  # [bs, n_sample, hsize2]
        pred_body = self.res_block3(x)  # [bs, n_sample, out_dim]

        if self.pred_cam:
            pred_betas, pred_cam = self.shape_cam_decoder(context)
            pred_cam = pred_cam.unsqueeze(1).repeat(1, num_samples, 1)
        else:
            pred_betas = self.shape_decoder(context)
            pred_cam = None
        pred_betas = pred_betas.unsqueeze(1).repeat(1, num_samples, 1)  # [bs, n_sample, 10]

        pred_body_coords, pred_smpl_params, pred_pose_6d = None, None, None
        if self.out_type == 'params':
            pred_pose_6d = pred_body.clone()
            pred_pose = rot6d_to_rotmat(pred_body.reshape(batch_size * num_samples, -1)).view(batch_size, num_samples, 24, 3, 3)  # [bs, num_sample, 24, 3, 3]
            pred_smpl_params = {'global_orient': pred_pose[:, :, [0]], 'body_pose': pred_pose[:, :, 1:], 'betas': pred_betas}
        else:
            pred_body_coords = pred_body.reshape(batch_size, num_samples, -1, 3)
        return pred_body_coords, pred_smpl_params, pred_pose_6d, pred_cam



