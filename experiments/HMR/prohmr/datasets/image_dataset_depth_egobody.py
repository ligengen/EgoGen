from os.path import join
import numpy as np
from typing import Dict 
from yacs.config import CfgNode
from os.path import join, exists, basename
import pickle as pkl
import torch
import cv2
import pyrender
from PIL import ImageDraw
import PIL.Image as pil_img
import trimesh
import os
import ast
import glob
from tqdm import tqdm
import smplx
import math
import pandas as pd
import json

from .dataset import Dataset
from .utils_depth_data import get_example
from ..utils.pose_utils import *
from ..utils.camera import create_camera
from .dataset_utils import get_right_full_img_pth, get_transf_matrices, parse_img_full_path
from ..utils.geometry import *

class ImageDatasetDepthEgoBody(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 split='train',
                 spacing=1,
                 # add_scale=1.0,
                 device=None,
                 do_augment=False,
                 data_source='real',
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDatasetDepthEgoBody, self).__init__()
        self.train = train
        self.split = split
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment

        # self.img_size = cfg.MODEL.IMAGE_SIZE
        # self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        # self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)


        self.img_dir = img_dir
        self.data = np.load(dataset_file)
        # with open(join(self.img_dir, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        #     self.transf_matrices = pkl.load(fp)

        self.imgname = self.data['imgname']

        # try:
        #     for id, x in enumerate(self.imgname):
        #         get_right_full_img_pth(x, self.img_dir)
        # except:
        #     print('[INFO] imgname is not in the right format, try to parse it.')
        #     print(id, x, self.img_dir)
        # import pdb; pdb.set_trace()
        if data_source == 'real':
            [self.imgname, self.seq_names, _] = zip(*[get_right_full_img_pth(x, self.img_dir) for x in self.imgname])   # absolute dir
        self.imgname = self.imgname[::spacing]


        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 22, 24, 23]  # for smplx 25 topology
        self.flip_3d_keypoint_permutation = body_permutation_3d

        self.body_pose = self.data['body_pose'].astype(np.float)[::spacing]  # [n_sample, 69]
        self.betas = self.data['betas'].astype(np.float)[::spacing]
        self.global_orient_depth = self.data['global_orient_depth'].astype(np.float)[::spacing]  # [n_sample, 3]
        self.transl_depth = self.data['transl_depth'].astype(np.float)[::spacing]
        self.keypoints_3d_depth = self.data['3d_joints_depth'].astype(np.float)[::spacing]
        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)


        # self.smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male')
        # self.smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female')

        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz', num_pca_comps=12,
                                  create_global_orient=True, create_body_pose=True, create_betas=True, create_transl=True,
                                  create_left_hand_pose=True, create_right_hand_pose=True,
                                  create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                  create_reye_pose=True, )  #.to(device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz', num_pca_comps=12,
                                  create_global_orient=True, create_body_pose=True, create_betas=True, create_transl=True,
                                  create_left_hand_pose=True, create_right_hand_pose=True,
                                  create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                  create_reye_pose=True, )  #.to(device)

        #  ????? way of creating smplx model

        self.dataset_len = len(self.imgname)
        print('[INFO] find {} samples in {}.'.format(self.dataset_len, dataset_file))




    # def get_transf_matrices_per_frame(self, img_name, seq_name):
    #
    #     transf_mtx_seq = self.transf_matrices[seq_name]
    #     kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
    #     holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix
    #
    #     timestamp = basename(img_name).split('_')[0]
    #     holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
    #     return kinect2holo, holo2pv



    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        image_file = join(self.img_dir, self.imgname[idx])  # absolute path
        keypoints_3d = self.keypoints_3d_depth[idx][0:25].copy()  # [25, 3], smplx joints

        # center = self.center[idx].copy().astype(np.float32)
        # center_x = center[0]
        # center_y = center[1]
        # bbox_size = self.scale[idx].astype(np.float32) * 200

        body_pose = self.body_pose[idx].copy().astype(np.float32)  # [63]
        betas = self.betas[idx].copy().astype(np.float32)  # [10]
        global_orient = self.global_orient_depth[idx].copy().astype(np.float32)  # 3
        transl = self.transl_depth[idx].copy().astype(np.float32)  # 3
        gender = self.gender[idx].copy()


        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }


        item = {}

        augm_config = self.cfg.DATASETS.CONFIG
        img, keypoints_3d_auge, smpl_params = get_example(image_file, keypoints_3d, smpl_params,
                                                          self.flip_3d_keypoint_permutation,
                                                          self.do_augment, augm_config,
                                                          smpl_male=self.smplx_male, smpl_female=self.smplx_female, gender=gender)

        item['img'] = img
        item['imgname'] = image_file  # '/mnt/ssd/egobody_release/egocentric_color/recording_20220415_S36_S35_01/2022-04-15-161202/PV/132945055822281630_frame_02030.jpg'
        item['keypoints_3d'] = keypoints_3d_auge.astype(np.float32)  # [25, 3]
        item['smpl_params'] = smpl_params
        for key in item['smpl_params'].keys():
            item['smpl_params'][key] = item['smpl_params'][key].astype(np.float32)
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['idx'] = idx
        item['gender'] = gender
        return item


class ImageDatasetDepthMix(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 real_dataset_file: str,
                 syn_dataset_file: str,
                 real_img_dir: str,
                 syn_img_dir: str,
                 train: bool = True,
                 split='train',
                 spacing=1,
                 # add_scale=1.0,
                 device=None,
                 do_augment=False,
                 data_source='real',
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDatasetDepthMix, self).__init__()
        self.train = train
        self.split = split
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment

        syn_data = np.load(syn_dataset_file)
        real_data = np.load(real_dataset_file)
        
        self.data = {}
        for key in ['body_pose', 'betas', 'global_orient_depth', 'transl_depth', '3d_joints_depth', 'gender']:
            self.data[key] = np.concatenate((syn_data[key], real_data[key]), axis=0)
            # print(key, self.data[key].shape, syn_data[key].shape, real_data[key].shape)

        syn_imgname = syn_data['imgname'][::spacing]
        syn_imgname = np.array([join(syn_img_dir, x) for x in syn_imgname])

        real_imgname = real_data['imgname']
        [real_imgname, _, _] = zip(*[get_right_full_img_pth(x, real_img_dir) for x in real_imgname]) 
        real_imgname = np.array([join(real_img_dir, x) for x in real_imgname])

        self.imgname = np.concatenate((syn_imgname, real_imgname), axis=0)
    

        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 22, 24, 23]  # for smplx 25 topology
        self.flip_3d_keypoint_permutation = body_permutation_3d

        self.body_pose = self.data['body_pose'].astype(np.float)[::spacing]  # [n_sample, 69]
        self.betas = self.data['betas'].astype(np.float)[::spacing]
        self.global_orient_depth = self.data['global_orient_depth'].astype(np.float)[::spacing]  # [n_sample, 3]
        self.transl_depth = self.data['transl_depth'].astype(np.float)[::spacing]
        self.keypoints_3d_depth = self.data['3d_joints_depth'].astype(np.float)[::spacing]
        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)

        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz', num_pca_comps=12,
                                  create_global_orient=True, create_body_pose=True, create_betas=True, create_transl=True,
                                  create_left_hand_pose=True, create_right_hand_pose=True,
                                  create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                  create_reye_pose=True, )  #.to(device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz', num_pca_comps=12,
                                  create_global_orient=True, create_body_pose=True, create_betas=True, create_transl=True,
                                  create_left_hand_pose=True, create_right_hand_pose=True,
                                  create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                  create_reye_pose=True, )  #.to(device)

        self.dataset_len = len(self.imgname)
        print('[INFO] find {} samples in real dataset {}.'.format(real_imgname.shape[0], real_dataset_file))
        print('[INFO] find {} samples in synthetic dataset{}.'.format(syn_imgname.shape[0], syn_dataset_file))
        print('[INFO] find {} samples in total.'.format(self.dataset_len))




    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        # image_file = join(self.img_dir, self.imgname[idx])  # absolute path
        image_file = self.imgname[idx]
        keypoints_3d = self.keypoints_3d_depth[idx][0:25].copy()  # [25, 3], smplx joints


        body_pose = self.body_pose[idx].copy().astype(np.float32)  # [63]
        betas = self.betas[idx].copy().astype(np.float32)  # [10]
        global_orient = self.global_orient_depth[idx].copy().astype(np.float32)  # 3
        transl = self.transl_depth[idx].copy().astype(np.float32)  # 3
        gender = self.gender[idx].copy()


        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }


        item = {}

        augm_config = self.cfg.DATASETS.CONFIG
        img, keypoints_3d_auge, smpl_params = get_example(image_file, keypoints_3d, smpl_params,
                                                          self.flip_3d_keypoint_permutation,
                                                          self.do_augment, augm_config,
                                                          smpl_male=self.smplx_male, smpl_female=self.smplx_female, gender=gender)

        item['img'] = img
        item['imgname'] = image_file  # '/mnt/ssd/egobody_release/egocentric_color/recording_20220415_S36_S35_01/2022-04-15-161202/PV/132945055822281630_frame_02030.jpg'
        item['keypoints_3d'] = keypoints_3d_auge.astype(np.float32)  # [25, 3]
        item['smpl_params'] = smpl_params
        for key in item['smpl_params'].keys():
            item['smpl_params'][key] = item['smpl_params'][key].astype(np.float32)
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['idx'] = idx
        item['gender'] = gender
        return item