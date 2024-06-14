import torch
import torch.nn as nn
import numpy as np
# import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from prohmr.models import SMPL
from yacs.config import CfgNode

import pyrender
from PIL import ImageDraw
import PIL.Image as pil_img
import smplx
import trimesh
import cv2
# import open3d as o3d

import torch.nn.functional as F

import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from prohmr.utils import SkeletonRenderer
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
# from prohmr.optimization import OptimizationTask
from .backbones import create_backbone
from .heads import SMPLFlow
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from ..utils.renderer import *


class ProHMREgobody(nn.Module):

    def __init__(self, cfg: CfgNode, device=None, writer=None, logger=None, with_focal_length=False, with_bbox_info=False, with_cam_center=False, with_vfov=False, with_joint_vis=False,
                 with_full_2d_loss=False, with_global_3d_loss=False, with_transl_loss=False):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super(ProHMREgobody, self).__init__()

        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.logger = logger

        self.with_focal_length = with_focal_length
        self.with_bbox_info = with_bbox_info
        self.with_cam_center = with_cam_center

        self.with_full_2d_loss = with_full_2d_loss
        self.with_global_3d_loss = with_global_3d_loss
        self.with_transl_loss = with_transl_loss

        self.with_vfov = with_vfov
        self.with_joint_vis = with_joint_vis
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg).to(self.device)

        # Create Normalizing Flow head
        contect_feats_dim = cfg.MODEL.FLOW.CONTEXT_FEATURES
        if self.with_focal_length or self.with_vfov:
            contect_feats_dim = contect_feats_dim + 1
        if self.with_bbox_info:
            contect_feats_dim = contect_feats_dim + 3
        if self.with_cam_center:
            contect_feats_dim = contect_feats_dim + 2
        if self.with_joint_vis:
            contect_feats_dim = contect_feats_dim + 25
        print('contect_feats_dim:', contect_feats_dim)
        self.flow = SMPLFlow(cfg, contect_feats_dim=contect_feats_dim).to(self.device)

        # Create discriminator
        self.discriminator = Discriminator().to(self.device)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg).to(self.device)

        self.smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male').to(self.device)
        self.smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female').to(self.device)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # # Setup renderer for visualization
        # self.renderer = SkeletonRenderer(self.cfg)
        # # Disable automatic optimization since we use adversarial training
        # self.automatic_optimization = False

        param_size = 0
        buffer_size = 0
        for param in self.backbone.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.backbone.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for param in self.flow.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.flow.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('model size: {:.3f}MB'.format(size_all_mb))




    def init_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        self.optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.flow.parameters()),
                                     lr=self.cfg.TRAIN.LR,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        self.optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        # return optimizer, optimizer_disc

    def initialize(self, batch: Dict, conditioning_feats: torch.Tensor):
        """
        Initialize ActNorm buffers by running a dummy forward step
        Args:
            batch (Dict): Dictionary containing batch data
            conditioning_feats (torch.Tensor): Tensor of shape (N, C) containing the conditioning features extracted using thee backbonee
        """
        # Get ground truth SMPL params, convert them to 6D and pass them to the flow module together with the conditioning feats.
        # Necessary to initialize ActNorm layers.
        smpl_params = {k: v.clone() for k,v in batch['smpl_params'].items()}
        batch_size = smpl_params['body_pose'].shape[0]
        has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)[has_smpl_params]
        conditioning_feats = conditioning_feats[has_smpl_params]
        with torch.no_grad():
            _, _ = self.flow.log_prob(smpl_params, conditioning_feats)
            self.initialized |= True


    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        if train:
            num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES
        else:
            num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES


        # Use RGB image as input
        x = batch['img']  # [bs, 3, 224, 224]
        batch_size = x.shape[0]

        # ####################################
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        # Compute keypoint features using the backbone
        conditioning_feats = self.backbone(x)  # [64, 2048]
        if self.with_focal_length:
            conditioning_feats = torch.cat([batch['fx'].unsqueeze(1), conditioning_feats], dim=-1)  # [bs, 1+2048]

        if self.with_bbox_info:
            orig_fx = batch['fx'] * self.cfg.CAM.FX_NORM_COEFF
            bbox_info = torch.stack([batch['box_center'][:, 0] / orig_fx, batch['box_center'][:, 1] / orig_fx, batch['box_size'] / orig_fx], dim=-1)  # [bs, 3]?
            conditioning_feats = torch.cat([bbox_info, conditioning_feats], dim=-1)  # [bs, 3(+1)+2048]
        if self.with_cam_center:
            orig_fx = batch['fx'] * self.cfg.CAM.FX_NORM_COEFF
            cam_center = torch.stack([batch['cam_cx'] / orig_fx, batch['cam_cy'] / orig_fx], dim=-1)  # [bs, 3]?
            conditioning_feats = torch.cat([cam_center, conditioning_feats], dim=-1)  # [bs, 2(+3)(+1)+2048]

        if self.with_vfov:
            conditioning_feats = torch.cat([batch['vx'].unsqueeze(1), conditioning_feats], dim=-1)
        if self.with_joint_vis:
            joint_vis_mask = batch['keypoints_2d_vis_mask']  # [bs, 25]
            conditioning_feats = torch.cat([joint_vis_mask, conditioning_feats], dim=-1)
        # conditioning_feats = torch.cat([batch['fx'].unsqueeze(1), batch['cx'].unsqueeze(1), batch['cy'].unsqueeze(1), conditioning_feats], dim=-1)  # [bs, 3+2048]


        # If ActNorm layers are not initialized, initialize them
        if not self.initialized.item():
            # import pdb; pdb.set_trace()
            self.initialize(batch, conditioning_feats)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            # pred_smpl_params: global_orient: [bs, num_sample-1, 1, 3, 3], body_pose: [bs, num_sample-1, 23, 3, 3], betas: [bs, 10]
            # pred_cam: [bs, 1, 3]
            # log_prob: [bs, 1]
            # pred_pose_6d: [bs, 1, 144]
            pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)  # [bs, num_sample-1, 3, 3]
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)  # [bs, 1, 144]
            pred_smpl_params_mode, pred_cam_mode, log_prob_mode, _,  pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            pred_smpl_params = {k: torch.cat((pred_smpl_params_mode[k], v), dim=1) for k,v in pred_smpl_params.items()}
            pred_cam = torch.cat((pred_cam_mode, pred_cam), dim=1)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
            pred_pose_6d = torch.cat((pred_pose_6d_mode, pred_pose_6d), dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params, pred_cam, log_prob, _,  pred_pose_6d = self.flow(conditioning_feats, z=z_0)

        # ????? meaning of the forward step (pred_smpl_params_mode,conditioning_feats)

        # ###################
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        # ###################

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam  # [bs, num_sample, 3]
        #  global_orient: [bs, num_sample, 1, 3, 3], body_pose: [bs, num_sample, 23, 3, 3], shape...
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['log_prob'] = log_prob.detach()  # [bs, 2]
        output['conditioning_feats'] = conditioning_feats
        output['pred_pose_6d'] = pred_pose_6d

        # ????? pred_pose_6d meaning??

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints  # [bs*num_sample, 45, 3]
        pred_vertices = smpl_output.vertices  # [bs*num_sample, 6890, 3]
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)  # [bs, num_sample, 24, 3]
        output['pred_vertices'] = pred_vertices.reshape(batch_size, num_samples, -1, 3)  # [bs, num_sample, 6890, 3]

        # Compute camera translation
        # device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        if self.with_focal_length or self.with_vfov:
            # focal_length = 0.5 * 1920 / torch.tan(0.5 * batch['vx']).unsqueeze(-1).unsqueeze(-1)
            focal_length = batch['fx'].unsqueeze(-1).unsqueeze(-1)  # [bs, 1, 1]
            focal_length = focal_length.repeat(1, num_samples, 2)  # [bs, n_sample, 2]
            focal_length = focal_length * self.cfg.CAM.FX_NORM_COEFF
            # todo: specify cam center or not?
            camera_center_full = torch.cat([batch['cam_cx'].unsqueeze(-1), batch['cam_cy'].unsqueeze(-1)], dim=-1).unsqueeze(1)  # [bs, 1, 2]
            camera_center_full = camera_center_full.repeat(1, num_samples, 1)  # [bs, 2, 2]
            # camera_center_full[:, :, 0] = camera_center_full[:, :, 0] * self.cfg.CAM.CX_NORM_COEFF
            # camera_center_full[:, :, 1] = camera_center_full[:, :, 1] * self.cfg.CAM.CY_NORM_COEFF
        else:
            focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, num_samples, 2, device=self.device, dtype=dtype)  # 5000
            camera_center_full = torch.tensor([[[960.0, 540.0]]]).to(self.device).float().repeat(batch_size, num_samples, 1)  # [bs, n_sample, 2]

        pred_cam_t = torch.stack([pred_cam[:, :, 1], pred_cam[:, :, 2],
                                  2 * focal_length[:, :, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, :, 0] + 1e-9)],
                                 dim=-1)
        output['pred_cam_t'] = pred_cam_t  # [bs, num_sample, 3]

        pred_cam_t = pred_cam_t.reshape(-1, 3)  # [bs*num_sample, 3]
        focal_length = focal_length.reshape(-1, 2)  # [bs*num_sample, 2]
        camera_center_full = camera_center_full.reshape(-1, 2)  # [bs*num_sample, 2]


        # if (self.with_focal_length or self.with_vfov) and self.with_full_2d_loss:
        if 1:
            pred_cam_t_full = convert_pare_to_full_img_cam(pare_cam=pred_cam.reshape(-1, 3), bbox_height=batch['box_size'].unsqueeze(1).repeat(1, num_samples).reshape(-1),
                                                           bbox_center=batch['box_center'].unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, 2),
                                                           # img_w=1920, img_h=1080,
                                                           img_w=camera_center_full[:, 0]*2, img_h=camera_center_full[:, 1]*2,
                                                           focal_length=focal_length[:, 0],
                                                           crop_res=self.cfg.MODEL.IMAGE_SIZE)  # [bs*num_sample, 3]
            # import pdb; pdb.set_trace()
            pred_keypoints_3d_full = output['pred_keypoints_3d'].reshape(batch_size*num_samples, -1, 3) + pred_cam_t_full.unsqueeze(1)
            output['pred_keypoints_3d_full'] = pred_keypoints_3d_full.reshape(batch_size, num_samples, -1, 3)
            pred_keypoints_2d_full = perspective_projection(pred_keypoints_3d,
                                                            translation=pred_cam_t_full,
                                                            camera_center=camera_center_full,              # todo: specify cam center or not?
                                                            focal_length=focal_length)  # [bs*n_sample, 45, 2]
            # import pdb; pdb.set_trace()
            pred_keypoints_2d_full[:, :, 0] = pred_keypoints_2d_full[:, :, 0] / 1920 - 0.5  # in [-0.5, 0.5]
            pred_keypoints_2d_full[:, :, 1] = pred_keypoints_2d_full[:, :, 1] / 1080 - 0.5  # in [-0.5, 0.5]
            # import pdb; pdb.set_trace()
            output['pred_keypoints_2d_full'] = pred_keypoints_2d_full.reshape(batch_size, num_samples, -1, 2)  # [bs, num_sample, n_smpl_joints, 2]
            output['pred_cam_t_full'] = pred_cam_t_full.reshape(batch_size, num_samples, -1)   # [bs, num_sample, 3]

        # on auge cropped img
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length)  # in [-224/2, 224/2], default cx=cy=0
        pred_keypoints_2d = pred_keypoints_2d / self.cfg.MODEL.IMAGE_SIZE  # in [-0.5, 0.5], default cx=cy=0, if cx=cy=IMAGE_SIZE/2, in [0,1] ?

        # print(pred_keypoints_2d[0])

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, num_samples, -1, 2)  # [bs, num_sample, n_smpl_joints, 2]
        # print(output['pred_keypoints_2d'][0][0][0:24])
        # import pdb; pdb.set_trace()
        return output
    
        ### ????? projection and the 2d loss

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        conditioning_feats = output['conditioning_feats']
        pred_keypoints_2d = output['pred_keypoints_2d']  # [bs, n_sample, 45, 2]
        pred_keypoints_3d = output['pred_keypoints_3d'][:, :, 0:24]  # [bs, n_sample, 24, 3]
        pred_keypoints_3d_full = output['pred_keypoints_3d_full'][:, :, 0:24]

        print(self.smpl.smpl_to_openpose)
        ### change smpl topology to openpose topology
        pred_keypoints_2d = pred_keypoints_2d[:, :, self.smpl.smpl_to_openpose, :]  # [bs, num_samples, 25, 2] todo: check if correct openpose topology
        # if (self.with_focal_length or self.with_vfov) and self.with_full_2d_loss:
        pred_keypoints_2d_full = output['pred_keypoints_2d_full']
        pred_keypoints_2d_full = pred_keypoints_2d_full[:, :, self.smpl.smpl_to_openpose, :]

        batch_size = pred_smpl_params['body_pose'].shape[0]
        num_samples = pred_smpl_params['body_pose'].shape[1]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_2d_full = batch['orig_keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']  # [bs, 24, 3]
        gt_keypoints_3d_full = batch['keypoints_3d_full']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_gender = batch['gender']

        # ################### debug: vis
        # images = batch['img']
        # images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        # images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        # images = 255 * images.permute(0, 2, 3, 1).cpu().numpy()
        #
        #
        # full_images = batch['orig_img']
        # full_images = full_images * torch.tensor([0.229, 0.224, 0.225], device=full_images.device).reshape(1, 3, 1, 1)
        # full_images = full_images + torch.tensor([0.485, 0.456, 0.406], device=full_images.device).reshape(1, 3, 1, 1)
        # full_images = 255 * full_images.permute(0, 2, 3, 1).cpu().numpy()
        # gt_keypoints_2d_full = batch['orig_keypoints_2d']
        # pred_keypoints_2d_full = output['pred_keypoints_2d_full']
        # pred_keypoints_2d_full = pred_keypoints_2d_full[:, :, self.smpl.smpl_to_openpose, :]
        #
        # # import pdb;
        # # pdb.set_trace()
        #
        # for i in range(0, 5, 2):
        #     crop_img = images[i]  # [224, 224, 3]?
        #
        #     pred_keypoints_2d_mode = pred_keypoints_2d[i, 0].detach().cpu().numpy()  # [25, 2]
        #     pred_keypoints_2d_mode = (pred_keypoints_2d_mode + 0.5) * 224
        #     gt_keypoints_2d_vis = gt_keypoints_2d[i].detach().cpu().numpy()  # [25, 3]
        #     gt_keypoints_2d_vis[:, :-1] = (gt_keypoints_2d_vis[:, :-1] + 0.5) * 224
        #
        #     full_img = full_images[i]
        #     pred_keypoints_2d_mode_full = pred_keypoints_2d_full[i, 0].detach().cpu().numpy()
        #     pred_keypoints_2d_mode_full[:, 0] = (pred_keypoints_2d_mode_full[:, 0] + 0.5) * 1920
        #     pred_keypoints_2d_mode_full[:, 1] = (pred_keypoints_2d_mode_full[:, 1] + 0.5) * 1080
        #     gt_keypoints_2d_vis_full = gt_keypoints_2d_full[i].detach().cpu().numpy()
        #     gt_keypoints_2d_vis_full[:, 0] = (gt_keypoints_2d_vis_full[:, 0] + 0.5) * 1920
        #     gt_keypoints_2d_vis_full[:, 1] = (gt_keypoints_2d_vis_full[:, 1] + 0.5) * 1080
        #
        #     # vis on crop img
        #     output_img = pil_img.fromarray((crop_img).astype(np.uint8))
        #     # vis on full img
        #     output_img = pil_img.fromarray((full_img).astype(np.uint8))
        #     pred_keypoints_2d_mode = pred_keypoints_2d_mode_full
        #     gt_keypoints_2d_vis = gt_keypoints_2d_vis_full
        #
        #     # import pdb; pdb.set_trace()
        #
        #
        #
        #     draw = ImageDraw.Draw(output_img)
        #     line_joint_indexs_openpose = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9],
        #                                   [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]
        #     for k in range(25):
        #         draw.ellipse((pred_keypoints_2d_mode[k][0] - 2, pred_keypoints_2d_mode[k][1] - 2,
        #                       pred_keypoints_2d_mode[k][0] + 2, pred_keypoints_2d_mode[k][1] + 2), fill=(0, 255, 0, 0))
        #     for index_pair in line_joint_indexs_openpose:
        #         draw.line(xy=[pred_keypoints_2d_mode[index_pair[0]][0], pred_keypoints_2d_mode[index_pair[0]][1],
        #                       pred_keypoints_2d_mode[index_pair[1]][0], pred_keypoints_2d_mode[index_pair[1]][1]],
        #                   fill=(0, 255, 0, 0), width=1)
        #     for k in range(25):
        #         draw.ellipse((gt_keypoints_2d_vis[k][0] - 2, gt_keypoints_2d_vis[k][1] - 2,
        #                       gt_keypoints_2d_vis[k][0] + 2, gt_keypoints_2d_vis[k][1] + 2), fill=(255, 0, 0, 0))
        #     for index_pair in line_joint_indexs_openpose:
        #         if gt_keypoints_2d_vis[index_pair[0]][2] > 0 and gt_keypoints_2d_vis[index_pair[1]][2] > 0:
        #             draw.line(xy=[gt_keypoints_2d_vis[index_pair[0]][0], gt_keypoints_2d_vis[index_pair[0]][1],
        #                           gt_keypoints_2d_vis[index_pair[1]][0], gt_keypoints_2d_vis[index_pair[1]][1]],
        #                       fill=(255, 0, 0, 0), width=1)
        #     output_img.show()
        #
        #     # if i >=15:
        #     import pdb; pdb.set_trace()
        #
        #     # # camera_center = np.array([cx, cy])
        #     # w, h = 224, 224
        #     # camera_translation = output['pred_cam'][i][0].detach().cpu().numpy()
        #     # camera_translation[0] *= -1.
        #     # camera_pose = np.eye(4)
        #     # camera_pose[:3, 3] = camera_translation
        #     # # camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        #     # light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        #     # base_color = (1.0, 193 / 255, 193 / 255, 1.0)
        #     # material = pyrender.MetallicRoughnessMaterial(
        #     #     metallicFactor=0.0,
        #     #     alphaMode='OPAQUE',
        #     #     baseColorFactor=base_color
        #     # )
        #     # camera_center = [w / 2., h / 2.]
        #     # camera = pyrender.camera.IntrinsicsCamera(
        #     #     fx=self.cfg.EXTRA.FOCAL_LENGTH, fy=self.cfg.EXTRA.FOCAL_LENGTH,
        #     #     cx=camera_center[0], cy=camera_center[1])
        #     #
        #     # smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male').to(self.device)
        #     # smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female').to(self.device)
        #     # if batch['gender'][i] == 0:
        #     #     smpl_model = smpl_male
        #     # else:
        #     #     smpl_model = smpl_female
        #     # body_params_dict_new = {}
        #     # body_params_dict_new['global_orient'] = pred_smpl_params['global_orient'][i:(i+1), 0].detach()  # [1, 1, 3, 3]
        #     # # body_params_dict_new['transl'] = pred_smpl_params['transl'][i:(i+1), 0].detach()
        #     # body_params_dict_new['body_pose'] = pred_smpl_params['body_pose'][i:(i+1), 0].detach()  # [1, 23, 3, 3]
        #     # body_params_dict_new['betas'] = pred_smpl_params['betas'][i:(i+1), 0].detach()  # [1, 10]
        #     # # import pdb; pdb.set_trace()
        #     # cur_vertices_pv = smpl_model(**body_params_dict_new, pose2rot=False).vertices.detach().cpu().numpy().squeeze()
        #     # body = trimesh.Trimesh(cur_vertices_pv, smpl_model.faces, process=False)
        #     # body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
        #     #
        #     # scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
        #     #                        ambient_light=(0.3, 0.3, 0.3))
        #     # scene.add(camera, pose=camera_pose)
        #     # scene.add(light, pose=camera_pose)
        #     # scene.add(body_mesh, 'mesh')
        #     # r = pyrender.OffscreenRenderer(viewport_width=w,
        #     #                                viewport_height=h,
        #     #                                point_size=1.0)
        #     # color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        #     #
        #     # color = color.astype(np.float32) / 255.0
        #     # alpha = 1.0  # set transparency in [0.0, 1.0]
        #     # color[:, :, -1] = color[:, :, -1] * alpha
        #     # color = pil_img.fromarray((color * 255).astype(np.uint8))
        #     # # output_img = pil_img.fromarray((output_img).astype(np.uint8))  # on original img?
        #     # output_img.paste(color, (0, 0), color)
        #     # output_img.show()



        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d.unsqueeze(1).repeat(1, num_samples, 1, 1), joints_to_ign=[1, 9, 12])  # [bs, n_sample]
        loss_keypoints_2d_full = self.keypoint_2d_loss(pred_keypoints_2d_full, gt_keypoints_2d_full.unsqueeze(1).repeat(1, num_samples, 1, 1), joints_to_ign=[1, 9, 12])  # [bs, n_sample]
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=0, pelvis_align=True)  # [bs, n_sample]
        loss_keypoints_3d_full = self.keypoint_3d_loss(pred_keypoints_3d_full, gt_keypoints_3d_full.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_align=False)

        loss_transl = F.l1_loss(output['pred_cam_t_full'], gt_smpl_params['transl'].unsqueeze(1).repeat(1, num_samples, 1), reduction='mean')

        ####### compute v2v loss
        gt_smpl_output = self.smpl_male(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices = gt_smpl_output.vertices  # smplx vertices
        gt_joints = gt_smpl_output.joints
        gt_smpl_output_female = self.smpl_female(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices_female = gt_smpl_output_female.vertices
        gt_joints_female = gt_smpl_output_female.joints
        gt_vertices[gt_gender == 1, :, :] = gt_vertices_female[gt_gender == 1, :, :]  # [bs, 6890, 3]
        gt_joints[gt_gender == 1, :, :] = gt_joints_female[gt_gender == 1, :, :]

        gt_vertices = gt_vertices.unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 6890, 3]
        gt_pelvis = gt_joints[:, [0], :].clone().unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 1, 3]
        pred_vertices = output['pred_vertices']  # [bs, num_sample, 6890, 3]
        loss_v2v = self.v2v_loss(pred_vertices - pred_keypoints_3d[:, :, [0], :].clone(), gt_vertices - gt_pelvis).mean(dim=(2, 3))  # [bs, n_sample]

        # ############### visualize
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #
        # gt_body_o3d = o3d.geometry.TriangleMesh()
        # gt_body_o3d.vertices = o3d.utility.Vector3dVector(gt_vertices[0, 0].detach().cpu().numpy())  # [6890, 3]
        # gt_body_o3d.triangles = o3d.utility.Vector3iVector(self.smpl_male.faces)
        # gt_body_o3d.compute_vertex_normals()
        # gt_body_o3d.paint_uniform_color([0, 0, 1.0])
        #
        # transformation = np.identity(4)
        # transformation[:3, 3] = gt_pelvis[0, 0].detach().cpu().numpy()
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        # sphere.paint_uniform_color([70 / 255, 130 / 255, 180 / 255])  # steel blue 70,130,180
        # sphere.compute_vertex_normals()
        # sphere.transform(transformation)
        #
        # pred_body_o3d = o3d.geometry.TriangleMesh()
        # pred_body_o3d.vertices = o3d.utility.Vector3dVector(pred_vertices[0, 0].detach().cpu().numpy())  # [6890, 3]
        # pred_body_o3d.triangles = o3d.utility.Vector3iVector(self.smpl_male.faces)
        # pred_body_o3d.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh_frame, sphere, pred_body_o3d, gt_body_o3d])
        #
        # gt_vertices_align = gt_vertices - gt_pelvis
        # gt_body_o3d = o3d.geometry.TriangleMesh()
        # gt_body_o3d.vertices = o3d.utility.Vector3dVector(gt_vertices_align[0, 0].detach().cpu().numpy())  # [6890, 3]
        # gt_body_o3d.triangles = o3d.utility.Vector3iVector(self.smpl_male.faces)
        # gt_body_o3d.compute_vertex_normals()
        # gt_body_o3d.paint_uniform_color([0, 0, 1.0])
        #
        # pred_vertices_align = pred_vertices - pred_keypoints_3d[:, :, [0], :].clone()
        # pred_body_o3d = o3d.geometry.TriangleMesh()
        # pred_body_o3d.vertices = o3d.utility.Vector3dVector(pred_vertices_align[0, 0].detach().cpu().numpy())  # [6890, 3]
        # pred_body_o3d.triangles = o3d.utility.Vector3iVector(self.smpl_male.faces)
        # pred_body_o3d.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh_frame, pred_body_o3d, gt_body_o3d])
        #
        # o3d.visualization.draw_geometries([sphere, mesh_frame, pred_body_o3d, gt_body_o3d])


        loss_v2v_mode = loss_v2v[:, [0]].mean()  # avg over batch, vertices
        if loss_v2v.shape[1] > 1:
            loss_v2v_exp = loss_v2v[:, 1:].mean()
        else:
            loss_v2v_exp = torch.tensor(0., device=device, dtype=dtype)

        # Compute loss on SMPL parameters
        # loss_smpl_params: keys: ['global_orient', 'body_pose', 'betas']
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            if k != 'transl':
                gt = gt_smpl_params[k].unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size * num_samples, -1, 3, 3)
                has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples)
                loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1), gt.reshape(batch_size, num_samples, -1), has_gt)


        # import pdb; pdb.set_trace()
        # Compute mode and expectation losses for 3D and 2D keypoints
        # The first item of the second dimension always corresponds to the mode
        loss_keypoints_2d_mode = loss_keypoints_2d[:, [0]].sum() / batch_size
        if loss_keypoints_2d.shape[1] > 1:
            loss_keypoints_2d_exp = loss_keypoints_2d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_2d_exp = torch.tensor(0., device=device, dtype=dtype)


        loss_keypoints_2d_full_mode = loss_keypoints_2d_full[:, [0]].sum() / batch_size
        if loss_keypoints_2d_full.shape[1] > 1:
            loss_keypoints_2d_full_exp = loss_keypoints_2d_full[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_2d_full_exp = torch.tensor(0., device=device, dtype=dtype)
        # import pdb; pdb.set_trace()

        loss_keypoints_3d_mode = loss_keypoints_3d[:, [0]].sum() / batch_size
        if loss_keypoints_3d.shape[1] > 1:
            loss_keypoints_3d_exp = loss_keypoints_3d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_3d_exp = torch.tensor(0., device=device, dtype=dtype)

        loss_keypoints_3d_full_mode = loss_keypoints_3d_full[:, [0]].sum() / batch_size
        if loss_keypoints_3d_full.shape[1] > 1:
            loss_keypoints_3d_full_exp = loss_keypoints_3d_full[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_3d_full_exp = torch.tensor(0., device=device, dtype=dtype)

        loss_smpl_params_mode = {k: v[:, [0]].sum() / batch_size for k,v in loss_smpl_params.items()}
        if loss_smpl_params['body_pose'].shape[1] > 1:
            loss_smpl_params_exp = {k: v[:, 1:].sum() / (batch_size * (num_samples - 1)) for k,v in loss_smpl_params.items()}
        else:
            loss_smpl_params_exp = {k: torch.tensor(0., device=device, dtype=dtype) for k,v in loss_smpl_params.items()}


        # Filter out images with corresponding SMPL parameter annotations
        # smpl_params = {k: v.clone() for k,v in gt_smpl_params.items()}
        smpl_params = {k: v.clone() for k, v in gt_smpl_params.items() if k!='transl'}
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)
        has_smpl_params = (batch['has_smpl_params']['body_pose'] > 0)
        smpl_params = {k: v[has_smpl_params] for k, v in smpl_params.items()}
        # Compute NLL loss
        # Add some noise to annotations at training time to prevent overfitting
        if train:
            smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}
        if smpl_params['body_pose'].shape[0] > 0:
            log_prob, _ = self.flow.log_prob(smpl_params, conditioning_feats[has_smpl_params])
        else:
            log_prob = torch.zeros(1, device=device, dtype=dtype)
        loss_nll = -log_prob.mean()


        # Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1)
        loss_pose_6d_mode = loss_pose_6d[:, 0].mean()
        loss_pose_6d_exp = loss_pose_6d[:, 1:].mean()

        # import pdb; pdb.set_trace()

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_EXP'] * loss_keypoints_3d_exp+ \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_FULL_EXP'] * loss_keypoints_3d_full_exp * self.with_global_3d_loss + \
               self.cfg.LOSS_WEIGHTS['V2V_EXP'] * loss_v2v_exp + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_EXP'] * loss_keypoints_2d_exp * (1-self.with_full_2d_loss) + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_FULL_EXP'] * loss_keypoints_2d_full_exp * self.with_full_2d_loss + \
               self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_exp+loss_pose_6d_mode)+\
               sum([loss_smpl_params_exp[k] * self.cfg.LOSS_WEIGHTS[(k+'_EXP').upper()] for k in loss_smpl_params_exp])+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode+ \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_FULL_MODE'] * loss_keypoints_3d_full_mode * self.with_global_3d_loss + \
               self.cfg.LOSS_WEIGHTS['TRANSL'] * loss_transl * self.with_transl_loss + \
               self.cfg.LOSS_WEIGHTS['V2V_MODE'] * loss_v2v_mode + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_MODE'] * loss_keypoints_2d_mode * (1-self.with_full_2d_loss) + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_FULL_MODE'] * loss_keypoints_2d_full_mode * self.with_full_2d_loss + \
               sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params_mode])

        losses = dict(loss=loss.detach(),
                      loss_nll=loss_nll.detach(),
                      loss_pose_6d_exp=loss_pose_6d_exp,
                      loss_pose_6d_mode=loss_pose_6d_mode,
                      loss_keypoints_2d_exp=loss_keypoints_2d_exp.detach(),
                      loss_keypoints_2d_full_exp=loss_keypoints_2d_full_exp.detach(),
                      loss_keypoints_3d_exp=loss_keypoints_3d_exp.detach(),
                      loss_keypoints_3d_full_exp=loss_keypoints_3d_full_exp.detach(),
                      loss_v2v_exp=loss_v2v_exp.detach(),
                      loss_keypoints_2d_mode=loss_keypoints_2d_mode.detach(),
                      loss_keypoints_2d_full_mode=loss_keypoints_2d_full_mode.detach(),
                      loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach(),
                      loss_keypoints_3d_full_mode=loss_keypoints_3d_full_mode.detach(),
                      loss_transl=loss_transl.detach(),
                      loss_v2v_mode=loss_v2v_mode.detach(),)

        # import pdb; pdb.set_trace()


        for k, v in loss_smpl_params_exp.items():
            losses['loss_' + k + '_exp'] = v.detach()
        for k, v in loss_smpl_params_mode.items():
            losses['loss_' + k + '_mode'] = v.detach()

        output['losses'] = losses

        return loss

    # def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
    #     """
    #     Log results to Tensorboard
    #     Args:
    #         batch (Dict): Dictionary containing batch data
    #         output (Dict): Dictionary containing the regression output
    #         step_count (int): Global training step count
    #         train (bool): Flag indicating whether it is training or validation mode
    #     """
    #
    #     mode = 'train' if train else 'val'
    #     summary_writer = self.logger.experiment
    #     batch_size = batch['keypoints_2d'].shape[0]
    #     images = batch['img']
    #     images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
    #     images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
    #     images = 255*images.permute(0, 2, 3, 1).cpu().numpy()
    #     num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES if mode == 'train' else self.cfg.TRAIN.NUM_TEST_SAMPLES
    #
    #     pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)
    #     gt_keypoints_3d = batch['keypoints_3d']
    #     gt_keypoints_2d = batch['keypoints_2d']
    #     losses = output['losses']
    #     pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, num_samples, 3)
    #     pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, num_samples, -1, 2)
    #
    #     for loss_name, val in losses.items():
    #         summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
    #     num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
    #     num_samples_per_image = min(num_samples, self.cfg.EXTRA.NUM_LOG_SAMPLES_PER_IMAGE)
    #
    #     gt_keypoints_3d = batch['keypoints_3d']
    #     pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)
    #
    #     # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
    #     predictions = self.renderer(pred_keypoints_3d[:num_images, :num_samples_per_image],
    #                                 gt_keypoints_3d[:num_images],
    #                                 2 * gt_keypoints_2d[:num_images],
    #                                 images=images[:num_images],
    #                                 camera_translation=pred_cam_t[:num_images, :num_samples_per_image])
    #     summary_writer.add_image('%s/predictions' % mode, predictions.transpose((2, 0, 1)), step_count)


    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step, [bs*n_sample, 207=23*3*3]
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        # batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']  # [bs, 69]
        gt_betas = batch['betas']
        batch_size = gt_body_pose.shape[0]
        # n_sample = body_pose.shape[0] // batch_size

        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)  # [bs, 23, 3, 3]
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())  # [bs*n_samples, 25]
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / disc_fake_out.shape[0]
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)  # [bs, 25]
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / disc_real_out.shape[0]
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        loss.backward()
        # self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, batch: Dict, mocap_batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        ### read input data
        # batch = joint_batch['img']   # [64, 3, 224, 224]
        # mocap_batch = joint_batch['mocap']
        # optimizer, optimizer_disc = self.optimizers(use_pl_optimizer=True)
        batch_size = batch['img'].shape[0]

        self.backbone.train()
        self.flow.train()
        # self.backbone.eval()
        # self.flow.eval()
        ### G forward step
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        pred_smpl_params = output['pred_smpl_params']
        ### compute G loss
        loss = self.compute_loss(batch, output, train=True)
        ### G adv loss
        disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1))
        loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size

        ### G backward
        loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        self.optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        self.optimizer.step()

        # import pdb; pdb.set_trace()
        ### D forward, backward
        loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1), self.optimizer_disc)

        output['losses']['loss_gen'] = loss_adv
        output['losses']['loss_disc'] = loss_disc

        # if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
        #     self.tensorboard_logging(batch, output, self.global_step, train=True)

        return output

    def validation_step(self, batch: Dict) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """

        self.backbone.eval()
        self.flow.eval()

        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        # pred_smpl_params = output['pred_smpl_params']
        # num_samples = pred_smpl_params['body_pose'].shape[1]
        loss = self.compute_loss(batch, output, train=False)
        # output['losses']: loss dict
        # self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    # def downstream_optimization(self, regression_output: Dict, batch: Dict, opt_task: OptimizationTask, **kwargs: Any) -> Dict:
    #     """
    #     Run downstream optimization using current regression output
    #     Args:
    #         regression_output (Dict): Dictionary containing batch data
    #         batch (Dict): Dictionary containing batch data
    #         opt_task (OptimizationTask): Class object for desired optimization task. Must implement __call__ method.
    #     Returns:
    #         Dict: Dictionary containing regression output.
    #     """
    #     conditioning_feats = regression_output['conditioning_feats']
    #     flow_net = lambda x: self.flow(conditioning_feats, z=x)
    #     return opt_task(flow_net=flow_net,
    #                     regression_output=regression_output,
    #                     data=batch,
    #                     **kwargs)
