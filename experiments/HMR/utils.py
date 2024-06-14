# import torchgeometry as tgm
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import datetime
import os, json, sys
import numpy as np
# from utils.Quaternions import Quaternions
# from utils.Pivots import Pivots
# import scipy.ndimage.filters as filters
import copy
from smplx.utils import SMPLOutput
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths


SMPL_EDGES = [(0, 1),
              [0, 2],
              [0, 3],
              [1, 4],
              [2, 5],
              [3, 6],
              [4, 7],
              [5, 8],
              [6, 9],
              [7, 10],
              [8, 11],
              [9, 12],
              [9, 13],
              [9, 14],
              [12, 15],
              [13, 16],
              [14, 17],
              [16, 18],
              [17, 19],
              [18, 20],
              [19, 21],
              [20, 22],
              [21, 23]]




LIMBS_BODY_SMPL = [(15, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         (20, 22),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         (21, 23),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11),]


LIMBS_MARKER = [(65, 63),
                     (65, 39),
                     (63, 9),
                     (39, 9),
                     (63, 64),
                     (65, 66),
                     (39, 56),
                     (9, 26),
                     (56, 1),
                     (26, 1),
                     (1, 61),
                     (61, 38),
                     (61, 8),
                     (38, 52),
                     (8, 22),
                     (52, 33),
                     (22, 3),
                     (33, 31),
                     (3, 31),
                     (33, 57),
                     (3, 27),
                     (57, 45),
                     (27, 14),
                     (45, 48),
                     (14, 18),
                     (48, 59),
                     (18, 29),
                     (59, 32),
                     (29, 2),
                     (32, 51),
                     (2, 21),
                     # arm
                     (56, 40),
                     (40, 43),
                     (43, 53),
                     (53, 42),
                     (26, 5),
                     (5, 10),
                     (10, 13),
                     (13, 23),
                     (23, 12),
                     ]


def eval_coll(pred_betas, pred_output, scene_pcd_verts, sample_idx, smpl_model):
    # pred_smpl_params = pred_output['pred_smpl_params']
    batch_size = pred_betas.shape[0]
    num_samples = pred_betas.shape[1]

    smpl_output_mode = SMPLOutput()
    smpl_output_mode.vertices = pred_output.vertices.reshape(batch_size, num_samples, -1, 3)[:, sample_idx]
    smpl_output_mode.joints = pred_output.joints.reshape(batch_size, num_samples, -1, 3)[:, sample_idx]
    smpl_output_mode.full_pose = pred_output.full_pose.reshape(batch_size, num_samples, -1, 3, 3)[:, sample_idx]  # [bs, 24, 3, 3]
    smpl_output_mode.full_pose = rotation_matrix_to_angle_axis(smpl_output_mode.full_pose.reshape(-1, 3, 3)).reshape(batch_size, -1)  # [bs, 24*3]

    smpl_output_mode_list = [SMPLOutput() for _ in range(batch_size)]
    coll_ratio_list = []
    # loss_coap_penetration_mode_list = torch.zeros([batch_size]).to(self.device)
    for i in range(batch_size):
        smpl_output_mode_list[i].vertices = smpl_output_mode.vertices[[i]].clone()
        smpl_output_mode_list[i].joints = smpl_output_mode.joints[[i]].clone()
        smpl_output_mode_list[i].full_pose = smpl_output_mode.full_pose[[i]].clone()
        ### sample scene verts
        bb_min = smpl_output_mode_list[i].vertices.min(1).values.reshape(1, 3).detach()
        bb_max = smpl_output_mode_list[i].vertices.max(1).values.reshape(1, 3).detach()
        # print(bb_min, bb_max)
        inds = (scene_pcd_verts[[i]] >= bb_min).all(-1) & (scene_pcd_verts[[i]] <= bb_max).all(-1)
        if inds.any():
            sampled_scene_pcd = scene_pcd_verts[[i]][inds].unsqueeze(0)  # [1, sample_verts_num, 3]
            occupancy = smpl_model.coap.query(sampled_scene_pcd, smpl_output_mode_list[i])  # [1, sample_verts_num] >0.5: inside, <0.5, outside
            # cur_coll_ratio = (occupancy>0.5).sum() / occupancy.shape[1]  # todo: how to evaluate?
            cur_coll_ratio = (occupancy > 0.5).sum() / scene_pcd_verts.shape[1]  # scene verts with collisions
            coll_ratio_list.append(cur_coll_ratio.detach().item())
        else:
            coll_ratio_list.append(0.0)  # no collision
    return coll_ratio_list