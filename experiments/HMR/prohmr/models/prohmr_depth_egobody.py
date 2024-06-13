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
from prohmr.models.backbones.resnet_depth import resnet
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis
# from prohmr.optimization import OptimizationTask
from .backbones import create_backbone
from .heads import SMPLXFlow
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from ..utils.renderer import *



class ProHMRDepthEgobody(nn.Module):

    def __init__(self, cfg: CfgNode, device=None, writer=None, logger=None, with_global_3d_loss=False):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super(ProHMRDepthEgobody, self).__init__()

        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.logger = logger

        self.with_global_3d_loss = with_global_3d_loss


        # self.backbone = create_backbone(cfg).to(self.device)
        self.backbone = resnet().to(self.device)

        # Create Normalizing Flow head
        contect_feats_dim = cfg.MODEL.FLOW.CONTEXT_FEATURES
        # print('contect_feats_dim:', contect_feats_dim)
        self.flow = SMPLXFlow(cfg, contect_feats_dim=contect_feats_dim).to(self.device)

        # Create discriminator
        self.discriminator = Discriminator().to(self.device)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = nn.MSELoss(reduction='none')

        # Instantiate SMPL model
        # smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        # self.smpl = SMPL(**smpl_cfg).to(self.device)
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', ext='npz').to(self.device)

        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz').to(self.device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz').to(self.device)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))


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
        # has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        
        # ????? aa to rotmat meaning?
        
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)
        # conditioning_feats = conditioning_feats[has_smpl_params]
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

        # ????? meaning of the num_samples

        # Use RGB image as input
        x = batch['img'].unsqueeze(1)  # [bs, 1, 224, 224]
        batch_size = x.shape[0]

        # Compute keypoint features using the backbone
        conditioning_feats = self.backbone(x)  # [bs, 2048]

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized.item():
            self.initialize(batch, conditioning_feats)

        # print(conditioning_feats.shape, num_samples)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            # pred_smpl_params: global_orient: [bs, num_sample-1, 1, 3, 3], body_pose: [bs, num_sample-1, 21, 3, 3], betas: [bs, 10]
            # pred_cam: [bs, 1, 3]
            # log_prob: [bs, 1]
            # pred_pose_6d: [bs, 1, 144]
            pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)  # [bs, num_sample-1, 3, 3]
            z_0 = torch.zeros(batch_size, 1, 22*6, device=x.device)  # [bs, 1, 132]
            pred_smpl_params_mode, pred_cam_mode, log_prob_mode, _,  pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            pred_smpl_params = {k: torch.cat((pred_smpl_params_mode[k], v), dim=1) for k,v in pred_smpl_params.items()}
            pred_cam = torch.cat((pred_cam_mode, pred_cam), dim=1)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
            pred_pose_6d = torch.cat((pred_pose_6d_mode, pred_pose_6d), dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params, pred_cam, log_prob, _,  pred_pose_6d = self.flow(conditioning_feats, z=z_0)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam  # [bs, num_sample, 3]
        #  global_orient: [bs, num_sample, 1, 3, 3], body_pose: [bs, num_sample, 23, 3, 3], shape...
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
        # for k, v in pred_smpl_params.items():
        #     print(k,v.shape)
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', ext='npz', batch_size=pred_smpl_params['global_orient'].shape[0]).to(self.device)
        smplx_output = self.smplx(**{k: v.float() for k,v in pred_smpl_params.items()})
        pred_keypoints_3d = smplx_output.joints  # [bs*num_sample, 127, 3]
        pred_vertices = smplx_output.vertices  # [bs*num_sample, 10475, 3]
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)  # [bs, num_sample, 127, 3]
        output['pred_vertices'] = pred_vertices.reshape(batch_size, num_samples, -1, 3)  # [bs, num_sample, 12475, 3]
        output['pred_keypoints_3d_global'] = output['pred_keypoints_3d'] + output['pred_cam'].unsqueeze(-2)  # [bs, n_sample, 127, 3]
        return output

        ### ????? rotation_matrix_to_angle_axis meaning?

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
        pred_pose_6d = output['pred_pose_6d']  # [bs, n_sample, 22*6]
        conditioning_feats = output['conditioning_feats']
        # pred_keypoints_3d = output['pred_keypoints_3d'][:, :, 0:25]  # [bs, n_sample, 25, 3]
        pred_keypoints_3d_global = output['pred_keypoints_3d_global'][:, :, 0:22]
        pred_keypoints_3d = output['pred_keypoints_3d'][:, :, 0:22]


        batch_size = pred_smpl_params['body_pose'].shape[0]
        num_samples = pred_smpl_params['body_pose'].shape[1]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        gt_keypoints_3d_global = batch['keypoints_3d'][:, 0:22]  # [bs, 22, 3]
        gt_smpl_params = batch['smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_gender = batch['gender']


        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d_global, gt_keypoints_3d_global.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=0, pelvis_align=True)  # [bs, n_sample]
        loss_keypoints_3d_full = self.keypoint_3d_loss(pred_keypoints_3d_global, gt_keypoints_3d_global.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_align=False)

        # loss_transl = F.l1_loss(output['pred_cam_t_full'], gt_smpl_params['transl'].unsqueeze(1).repeat(1, num_samples, 1), reduction='mean')

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
        gt_vertices[gt_gender == 1, :, :] = gt_vertices_female[gt_gender == 1, :, :]  # [bs, 10475, 3]
        gt_joints[gt_gender == 1, :, :] = gt_joints_female[gt_gender == 1, :, :]

        gt_vertices = gt_vertices.unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 10475, 3]
        gt_pelvis = gt_joints[:, [0], :].clone().unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 1, 3]
        pred_vertices = output['pred_vertices']  # [bs, num_sample, 10475, 3]
        loss_v2v = self.v2v_loss(pred_vertices - pred_keypoints_3d[:, :, [0], :].clone(), gt_vertices - gt_pelvis).mean(dim=(2, 3))  # [bs, n_sample]

        # ############### visualize
        # import open3d as o3d
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #
        # gt_body_o3d = o3d.geometry.TriangleMesh()
        # gt_body_o3d.vertices = o3d.utility.Vector3dVector(gt_vertices[0, 0].detach().cpu().numpy())  # [6890, 3]
        # gt_body_o3d.triangles = o3d.utility.Vector3iVector(self.smplx_male.faces)
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
        # pred_body_o3d.triangles = o3d.utility.Vector3iVector(self.smplx_male.faces)
        # pred_body_o3d.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh_frame, sphere, pred_body_o3d, gt_body_o3d])
        #
        # gt_vertices_align = gt_vertices - gt_pelvis
        # gt_body_o3d = o3d.geometry.TriangleMesh()
        # gt_body_o3d.vertices = o3d.utility.Vector3dVector(gt_vertices_align[0, 0].detach().cpu().numpy())  # [6890, 3]
        # gt_body_o3d.triangles = o3d.utility.Vector3iVector(self.smplx_male.faces)
        # gt_body_o3d.compute_vertex_normals()
        # gt_body_o3d.paint_uniform_color([0, 0, 1.0])
        #
        # pred_vertices_align = pred_vertices - pred_keypoints_3d[:, :, [0], :].clone()
        # pred_body_o3d = o3d.geometry.TriangleMesh()
        # pred_body_o3d.vertices = o3d.utility.Vector3dVector(pred_vertices_align[0, 0].detach().cpu().numpy())  # [6890, 3]
        # pred_body_o3d.triangles = o3d.utility.Vector3iVector(self.smplx_male.faces)
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
                # has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples)
                loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1), gt.reshape(batch_size, num_samples, -1))



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
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)  # [bs, 1,126]
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)
        # has_smpl_params = (batch['has_smpl_params']['body_pose'] > 0)
        smpl_params = {k: v for k, v in smpl_params.items()}
        # Compute NLL loss
        # Add some noise to annotations at training time to prevent overfitting
        if train:
            smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}
        if smpl_params['body_pose'].shape[0] > 0:
            log_prob, _ = self.flow.log_prob(smpl_params, conditioning_feats)
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
               self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_exp+loss_pose_6d_mode)+\
               sum([loss_smpl_params_exp[k] * self.cfg.LOSS_WEIGHTS[(k+'_EXP').upper()] for k in loss_smpl_params_exp])+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode+ \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_FULL_MODE'] * loss_keypoints_3d_full_mode * self.with_global_3d_loss + \
               self.cfg.LOSS_WEIGHTS['V2V_MODE'] * loss_v2v_mode + \
               sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params_mode])

        losses = dict(loss=loss.detach(),
                      loss_nll=loss_nll.detach(),
                      loss_pose_6d_exp=loss_pose_6d_exp,
                      loss_pose_6d_mode=loss_pose_6d_mode,
                      loss_keypoints_3d_exp=loss_keypoints_3d_exp.detach(),
                      loss_keypoints_3d_full_exp=loss_keypoints_3d_full_exp.detach(),
                      loss_v2v_exp=loss_v2v_exp.detach(),
                      loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach(),
                      loss_keypoints_3d_full_mode=loss_keypoints_3d_full_mode.detach(),
                      loss_v2v_mode=loss_v2v_mode.detach(),)

        # import pdb; pdb.set_trace()


        for k, v in loss_smpl_params_exp.items():
            losses['loss_' + k + '_exp'] = v.detach()
        for k, v in loss_smpl_params_mode.items():
            losses['loss_' + k + '_mode'] = v.detach()

        output['losses'] = losses

        return loss



    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
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
        ### compute G loss
        loss = self.compute_loss(batch, output, train=True)
        # ### G adv loss
        # disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1))
        # loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
        #
        # ### G backward
        # loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        self.optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        self.optimizer.step()

        # # import pdb; pdb.set_trace()
        # ### D forward, backward
        # loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1), pred_smpl_params['betas'].reshape(batch_size * num_samples, -1), self.optimizer_disc)
        #
        # output['losses']['loss_gen'] = loss_adv
        # output['losses']['loss_disc'] = loss_disc

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

        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        return output

