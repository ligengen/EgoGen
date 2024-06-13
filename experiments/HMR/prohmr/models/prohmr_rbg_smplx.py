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
from .heads import SMPLXFlow
from .discriminatorSmplx import DiscriminatorSmplx
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from ..utils.renderer import *
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis


class ProHMRRGBSmplx(nn.Module):

    def __init__(self, cfg: CfgNode, device=None, writer=None, logger=None, with_focal_length=False, with_bbox_info=False, with_cam_center=False, with_vfov=False, with_joint_vis=False,
                 with_full_2d_loss=False, with_global_3d_loss=False, with_transl_loss=False):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super(ProHMRRGBSmplx, self).__init__()

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
        self.flow = SMPLXFlow(cfg, contect_feats_dim=contect_feats_dim).to(self.device)

        # Create discriminator
        self.discriminator = DiscriminatorSmplx().to(self.device)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPLX model
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', batch_size= 144, ext='npz').to(self.device)
        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', batch_size= 36,  ext='npz').to(self.device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', batch_size = 36, ext='npz').to(self.device)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))

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

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized.item():
            self.initialize(batch, conditioning_feats)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            # pred_smpl_params: global_orient: [bs, num_sample-1, 1, 3, 3], body_pose: [bs, num_sample-1, 21, 3, 3], betas: [bs, 10]
            # pred_cam: [bs, 1, 3]
            # log_prob: [bs, 1]
            # pred_pose_6d: [bs, 1, 144]
            pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)  # [bs, num_sample-1, 3, 3]
            z_0 = torch.zeros(batch_size, 1, 22*6, device=x.device)  # [bs, 1, 144]
            pred_smpl_params_mode, pred_cam_mode, log_prob_mode, _,  pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            pred_smpl_params = {k: torch.cat((pred_smpl_params_mode[k], v), dim=1) for k,v in pred_smpl_params.items()}
            pred_cam = torch.cat((pred_cam_mode, pred_cam), dim=1)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
            pred_pose_6d = torch.cat((pred_pose_6d_mode, pred_pose_6d), dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, 22*6, device=x.device)
            pred_smpl_params, pred_cam, log_prob, _,  pred_pose_6d = self.flow(conditioning_feats, z=z_0)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam  # [bs, num_sample, 3]
        #  global_orient: [bs, num_sample, 1, 3, 3], body_pose: [bs, num_sample, 21, 3, 3], shape...
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['log_prob'] = log_prob.detach()  # [bs, 2]
        output['conditioning_feats'] = conditioning_feats
        output['pred_pose_6d'] = pred_pose_6d

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['global_orient'] = rotation_matrix_to_angle_axis(pred_smpl_params['global_orient'].reshape(-1, 3, 3)).reshape(batch_size * num_samples, -1, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['body_pose'] = rotation_matrix_to_angle_axis(pred_smpl_params['body_pose'].reshape(-1, 3, 3)).reshape(batch_size * num_samples, -1, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', ext='npz', batch_size=pred_smpl_params['global_orient'].shape[0]).to(self.device)
        smplx_output = self.smplx(**{k: v.float() for k,v in pred_smpl_params.items()})
        pred_keypoints_3d = smplx_output.joints  # [bs*num_sample, 45, 3]
        pred_vertices = smplx_output.vertices  # [bs*num_sample, 6890, 3]
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

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, num_samples, -1, 2)  # [bs, num_sample, n_smpl_joints, 2]

        return output

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
        pred_keypoints_3d = output['pred_keypoints_3d'][:, :, 0:22]  # [bs, n_sample, 24, 3]
        pred_keypoints_3d_full = output['pred_keypoints_3d_full'][:, :, 0:22]

        #print the attributes of smplx
        # for key in self.smplx.__dict__.keys():
        #     print(key)

        # smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        smplx_to_openpose = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
        ### change smpl topology to openpose topology
        pred_keypoints_2d = pred_keypoints_2d[:, :, smplx_to_openpose, :]  # [bs, num_samples, 25, 2] todo: check if correct openpose topology
        # if (self.with_focal_length or self.with_vfov) and self.with_full_2d_loss:
        pred_keypoints_2d_full = output['pred_keypoints_2d_full']
        pred_keypoints_2d_full = pred_keypoints_2d_full[:, :, smplx_to_openpose, :]


        batch_size = pred_smpl_params['body_pose'].shape[0]
        num_samples = pred_smpl_params['body_pose'].shape[1]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_2d_full = batch['orig_keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d'][:, 0:22]  # [bs, 22, 3]
        gt_keypoints_3d_full = batch['keypoints_3d_full'][:, 0:22]
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_gender = batch['gender']



        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d.unsqueeze(1).repeat(1, num_samples, 1, 1), joints_to_ign=[1, 9, 12])  # [bs, n_sample]
        loss_keypoints_2d_full = self.keypoint_2d_loss(pred_keypoints_2d_full, gt_keypoints_2d_full.unsqueeze(1).repeat(1, num_samples, 1, 1), joints_to_ign=[1, 9, 12])  # [bs, n_sample]
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=0, pelvis_align=True)  # [bs, n_sample]
        loss_keypoints_3d_full = self.keypoint_3d_loss(pred_keypoints_3d_full, gt_keypoints_3d_full.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_align=False)

        loss_transl = F.l1_loss(output['pred_cam_t_full'], gt_smpl_params['transl'].unsqueeze(1).repeat(1, num_samples, 1), reduction='mean')

        ####### compute v2v loss
        temp_bs = gt_smpl_params['body_pose'].shape[0]
        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz', batch_size=temp_bs).to(self.device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz', batch_size=temp_bs).to(self.device)
        
        gt_smpl_output = self.smplx_male(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices = gt_smpl_output.vertices  # smplx vertices
        gt_joints = gt_smpl_output.joints
        gt_smpl_output_female = self.smplx_female(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices_female = gt_smpl_output_female.vertices
        gt_joints_female = gt_smpl_output_female.joints
        gt_vertices[gt_gender == 1, :, :] = gt_vertices_female[gt_gender == 1, :, :]  # [bs, 6890, 3]
        gt_joints[gt_gender == 1, :, :] = gt_joints_female[gt_gender == 1, :, :]

        gt_vertices = gt_vertices.unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 6890, 3]
        gt_pelvis = gt_joints[:, [0], :].clone().unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 1, 3]
        pred_vertices = output['pred_vertices']  # [bs, num_sample, 6890, 3]
        loss_v2v = self.v2v_loss(pred_vertices - pred_keypoints_3d[:, :, [0], :].clone(), gt_vertices - gt_pelvis).mean(dim=(2, 3))  # [bs, n_sample]


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
    
    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)


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
        self.backbone.train()
        self.flow.train()

        # print("   ^^^^^^^^^^^^^^^ training_step  ^^^^^^^^^^^^^^^^")
        # output = self.debug_forward(batch, train=True)
        # loss = self.debug_loss(batch, output, train=True)
        # print("   ^^^^^^^^^^^^^^^ training_step  ^^^^^^^^^^^^^^^^")
        
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)

        # import pdb; pdb.set_trace()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

        
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)

        return output