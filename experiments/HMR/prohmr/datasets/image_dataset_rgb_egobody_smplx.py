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

from .dataset import Dataset
from .utils import get_example
from ..utils.pose_utils import *
from ..utils.camera import create_camera
from .dataset_utils import get_right_full_img_pth, get_transf_matrices, parse_img_full_path
import albumentations as A

class ImageDatasetEgoBodyRgbSmplx(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 spacing=1,
                 add_scale=1.0,
                 device=None,
                 do_augment=False,
                 data_source='real',
                 is_train=True,
                 is_aug = False,
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDatasetEgoBodyRgbSmplx, self).__init__()
        self.train = train
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment
        self.is_train =is_train
        self.is_aug = is_aug

        # print if do augmentation
        # self.do_augment = False
        if self.do_augment:
            print('[INFO] do augmentation in dataset')
        else :
            print('[INFO] do not augmentation in dataset')

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.fx_norm_coeff = self.cfg.CAM.FX_NORM_COEFF
        self.fy_norm_coeff = self.cfg.CAM.FY_NORM_COEFF
        self.cx_norm_coeff = self.cfg.CAM.CX_NORM_COEFF
        self.cy_norm_coeff = self.cfg.CAM.CY_NORM_COEFF

        self.img_dir = img_dir
        self.data = np.load(dataset_file)

        self.data_source = data_source


        # imgname = self.data['imgname'].copy()
        # for i in range(len(imgname)):
        #     relative_img_path = imgname[i].split('egobody_release/')[1]
        #     imgname[i] = self.img_dir + relative_img_path
        # self.data['imgname'] = imgname

        # bluir_imgname = self.data['blurimgname'].copy()
        # for i in range(len(bluir_imgname)):
        #     relative_img_path = bluir_imgname[i].split('egobody_release/')[1]
        #     bluir_imgname[i] = self.img_dir + relative_img_path
        # self.data['blurimgname'] = bluir_imgname
        
        if data_source != 'real':
            self.imgname = self.data['blurimgname']
            # bluir_imgname = self.data['blurimgname'].copy()
            # for i in range(len(bluir_imgname)):
            #     relative_img_path = bluir_imgname[i].split('egobody_release/')[1]
            #     bluir_imgname[i] = self.img_dir + relative_img_path
            # self.imgname = bluir_imgname
            # import pdb; pdb.set_trace()
            # [self.imgname, _, _] = zip(*[get_right_full_img_pth(x, self.img_dir) for x in self.imgname])
            self.imgname = self.imgname[::spacing]
        else:
            with open(join(self.img_dir, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
                self.transf_matrices = pkl.load(fp)

            self.imgname = self.data['imgname']
            # imgname = self.data['imgname'].copy()
            # for i in range(len(imgname)):
            #     relative_img_path = imgname[i].split('egobody_release/')[1]
            #     imgname[i] = self.img_dir + relative_img_path
            # self.imgname = imgname
            [self.imgname, self.seq_names, _] = zip(*[get_right_full_img_pth(x, self.img_dir) for x in self.imgname])   # absolute dir
            self.seq_names = [basename(x) for x in self.seq_names][::spacing]
            self.imgname = self.imgname[::spacing]

        # import pdb; pdb.set_trace()

        body_permutation_2d = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]  # for openpose 25 topology
        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 22, 24, 23]  # for smplx 25 topology
        # extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        # flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        self.flip_2d_keypoint_permutation = body_permutation_2d
        self.flip_3d_keypoint_permutation = body_permutation_3d

        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center'][::spacing]
        self.scale = self.data['scale'][::spacing] * add_scale
        # self.scale = self.data['scale'].reshape(len(self.center), -1).max(axis=-1) / 200.0  # todo: ?

        self.has_smpl = np.ones(len(self.imgname))
        # self.global_orient_kinect = self.data['global_orient'].astype(np.float)[::spacing]
        # self.transl_kinect = self.data['transl'].astype(np.float)[::spacing]
        self.body_pose = self.data['pose'].astype(np.float)[::spacing]
        self.betas = self.data['shape'].astype(np.float)[::spacing] 
        
        self.global_orient_pv = self.data['global_orient_pv'].astype(np.float)[::spacing]
        self.transl_pv = self.data['transl_pv'].astype(np.float)[::spacing]
        
        self.cx = self.data['cx'].astype(np.float)[::spacing]
        self.cy = self.data['cy'].astype(np.float)[::spacing]
        self.fx = self.data['fx'].astype(np.float)[::spacing]
        self.fy = self.data['fy'].astype(np.float)[::spacing]

        keypoints_openpose = self.data['valid_keypoints'][::spacing]

        self.keypoints_2d = keypoints_openpose
        self.keypoints_3d_pv = self.data['3d_joints_pv'].astype(np.float)[::spacing]

        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        self.length = self.scale.shape[0]

        self.dataset_len = len(self.imgname)
        print('[INFO] find {} samples in {}.'.format(self.dataset_len, dataset_file))


    def get_transf_matrices_per_frame(self, img_name, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix

        timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        image_file = join(self.img_dir, self.imgname[idx])
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d_pv[idx][0:25].copy()

        center = self.center[idx].copy().astype(np.float32)
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx].astype(np.float32) * 200
        body_pose = self.body_pose[idx].copy().astype(np.float32)  # 69? todo
        betas = self.betas[idx].copy().astype(np.float32)  # [10]
        global_orient = self.global_orient_pv[idx].copy().astype(np.float32)  # 3
        transl = self.transl_pv[idx].copy().astype(np.float32)  # 3
        gender = self.gender[idx].copy()

        fx = self.fx[idx].copy()
        fy = self.fy[idx].copy()
        cx = self.cx[idx].copy()
        cy = self.cy[idx].copy()

        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }

        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }

        # is_train_synthetic = self.is_train and self.data_source != 'real'
        is_train_synthetic = self.is_train and self.is_aug

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        # todo: how to augment fx, fy, cx, cy?
        img_patch, keypoints_2d, keypoints_2d_crop_vis_mask, keypoints_3d_crop_auge, keypoints_3d_full_auge, \
        smpl_params, has_smpl_params, img_size, img_patch_cv, \
        center_x_auge, center_y_auge, cam_cx_auge, auge_scale, keypoints_2d_aug_orig, rotated_img, rotated_cvimg = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    self.flip_2d_keypoint_permutation,
                                                                                                                  self.flip_3d_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.do_augment, augm_config, fx, cam_cx=cx, cam_cy=cy,
                                                                                                    is_train_synthetic=is_train_synthetic)


        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        if self.data_source != 'real':
            item['transf_kinect2holo'], item['transf_holo2pv'] = np.eye(4), np.eye(4)
        else:
            seq_name = self.seq_names[idx]
            item['transf_kinect2holo'], item['transf_holo2pv'] = self.get_transf_matrices_per_frame(image_file, seq_name)

        item['img'] = img_patch
        item['imgname'] = image_file  # '/mnt/ssd/egobody_release/egocentric_color/recording_20220415_S36_S35_01/2022-04-15-161202/PV/132945055822281630_frame_02030.jpg'
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)  # [25, 3]
        item['keypoints_3d'] = keypoints_3d_crop_auge.astype(np.float32)  # [24, 3]
        item['keypoints_3d_full'] = keypoints_3d_full_auge.astype(np.float32)
        item['keypoints_2d_vis_mask'] = keypoints_2d_crop_vis_mask  # [25] vis mask for openpose joint in augmented cropped img
        item['img_size'] = 1.0 * img_size[::-1].copy()  # array([1080., 1920.])
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['idx'] = idx
        item['gender'] = gender

        item['fx'] = (fx / self.fx_norm_coeff).astype(np.float32)
        item['fy'] = (fy / self.fy_norm_coeff).astype(np.float32)
        item['cam_cx'] = cam_cx_auge.astype(np.float32)
        item['cam_cy'] = cy.astype(np.float32)

        item['vx'] = 2 * math.atan(img_size[1] / (2 * fx))


        # augmented
        item['orig_keypoints_2d'] = keypoints_2d_aug_orig.astype(np.float32)
        item['box_center'] = np.array([center_x_auge, center_y_auge]).astype(np.float32)
        item['box_size'] = (bbox_size * auge_scale).astype(np.float32)
        item['orig_img'] = rotated_img  # original img rotate around (center_x_auge, center_y_auge)


        # import pdb;  pdb.set_trace()
        return item


class ImageDatasetEgoBodyRgbSmplxMix(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 real_dataset_file: str,
                 syn_dataset_file: str,
                 real_img_dir: str,
                 syn_img_dir: str,
                 train: bool = True,
                 spacing=1,
                 add_scale=1.0,
                 device=None,
                 do_augment=False,
                 data_source='real',
                 is_train=True,
                 is_aug = False,
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDatasetEgoBodyRgbSmplxMix, self).__init__()
        self.train = train
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment
        self.is_train =is_train
        self.is_aug = is_aug

        if self.do_augment:
            print('[INFO] do augmentation in dataset')
        else :
            print('[INFO] do not augmentation in dataset')

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.fx_norm_coeff = self.cfg.CAM.FX_NORM_COEFF
        self.fy_norm_coeff = self.cfg.CAM.FY_NORM_COEFF
        self.cx_norm_coeff = self.cfg.CAM.CX_NORM_COEFF
        self.cy_norm_coeff = self.cfg.CAM.CY_NORM_COEFF


        syn_data = np.load(syn_dataset_file)
        real_data = np.load(real_dataset_file)
        
        self.data = {}
        for key in ['center', 'scale', 'pose', 'shape', 'global_orient_pv', 'transl_pv', 'cx', 'cy', 'fx', 'fy', 'valid_keypoints', '3d_joints_pv', 'gender']:
            self.data[key] = np.concatenate((syn_data[key], real_data[key]), axis=0)
            # print(key, self.data[key].shape, syn_data[key].shape, real_data[key].shape)

        syn_imgname = syn_data['blurimgname'][::spacing]
        syn_imgname = np.array([join(syn_img_dir, x) for x in syn_imgname])
        syn_transf_kinect2holo = np.array([np.eye(4) for _ in range(syn_imgname.shape[0])])
        syn_transf_holo2pv = np.array([np.eye(4) for _ in range(syn_imgname.shape[0])])

        with open(join(real_img_dir, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
            self.transf_matrices = pkl.load(fp)

        real_imgname = real_data['imgname']
        real_imgname = ['/'.join(i.split('/')[-5:]) for i in real_imgname]
        [real_imgname, real_seqname, _] = zip(*[get_right_full_img_pth(x, real_img_dir) for x in real_imgname]) 
        real_seqname = [basename(x) for x in real_seqname]
        real_imgname = np.array([join(real_img_dir, x) for x in real_imgname])
        [real_transf_kinect2holo, real_transf_holo2pv] = zip(*[self.get_transf_matrices_per_frame(x, y) for x, y in zip(real_imgname, real_seqname)]) 
        real_transf_kinect2holo = np.array(real_transf_kinect2holo)
        real_transf_holo2pv = np.array(real_transf_holo2pv)

        self.transf_kinect2holo = np.concatenate((syn_transf_kinect2holo, real_transf_kinect2holo), axis=0)
        self.transf_holo2pv = np.concatenate((syn_transf_holo2pv, real_transf_holo2pv), axis=0)
        self.imgname = np.concatenate((syn_imgname, real_imgname), axis=0)

        # import pdb; pdb.set_trace()

        # if data_source != 'real':
        #     self.imgname = self.data['blurimgname']
        #     self.imgname = self.imgname[::spacing]
        # else:
        #     with open(join(self.img_dir, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        #         self.transf_matrices = pkl.load(fp)
        #     self.imgname = self.data['imgname']
        #     [self.imgname, self.seq_names, _] = zip(*[get_right_full_img_pth(x, self.img_dir) for x in self.imgname])   # absolute dir
        #     self.seq_names = [basename(x) for x in self.seq_names][::spacing]
        #     self.imgname = self.imgname[::spacing]

        body_permutation_2d = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]  # for openpose 25 topology
        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 22, 24, 23]  # for smplx 25 topology
 
        self.flip_2d_keypoint_permutation = body_permutation_2d
        self.flip_3d_keypoint_permutation = body_permutation_3d

        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center'][::spacing]
        self.scale = self.data['scale'][::spacing] * add_scale
        # self.scale = self.data['scale'].reshape(len(self.center), -1).max(axis=-1) / 200.0  # todo: ?

        self.has_smpl = np.ones(len(self.imgname))
        # self.global_orient_kinect = self.data['global_orient'].astype(np.float)[::spacing]
        # self.transl_kinect = self.data['transl'].astype(np.float)[::spacing]
        self.body_pose = self.data['pose'].astype(np.float)[::spacing]
        self.betas = self.data['shape'].astype(np.float)[::spacing] 
        
        self.global_orient_pv = self.data['global_orient_pv'].astype(np.float)[::spacing]
        self.transl_pv = self.data['transl_pv'].astype(np.float)[::spacing]
        
        self.cx = self.data['cx'].astype(np.float)[::spacing]
        self.cy = self.data['cy'].astype(np.float)[::spacing]
        self.fx = self.data['fx'].astype(np.float)[::spacing]
        self.fy = self.data['fy'].astype(np.float)[::spacing]

        keypoints_openpose = self.data['valid_keypoints'][::spacing]

        self.keypoints_2d = keypoints_openpose
        self.keypoints_3d_pv = self.data['3d_joints_pv'].astype(np.float)[::spacing]

        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        self.length = self.scale.shape[0]

        self.dataset_len = len(self.imgname)

        print('[INFO] find {} samples in real dataset {}.'.format(real_imgname.shape[0], real_dataset_file))
        print('[INFO] find {} samples in synthetic dataset{}.'.format(syn_imgname.shape[0], syn_dataset_file))
        print('[INFO] find {} samples in total.'.format(self.dataset_len))


    def get_transf_matrices_per_frame(self, img_name, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix

        timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        # image_file = join(self.img_dir, self.imgname[idx])
        image_file = self.imgname[idx]
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d_pv[idx][0:25].copy()

        center = self.center[idx].copy().astype(np.float32)
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx].astype(np.float32) * 200
        body_pose = self.body_pose[idx].copy().astype(np.float32)  # 69? todo
        betas = self.betas[idx].copy().astype(np.float32)  # [10]
        global_orient = self.global_orient_pv[idx].copy().astype(np.float32)  # 3
        transl = self.transl_pv[idx].copy().astype(np.float32)  # 3
        gender = self.gender[idx].copy()

        fx = self.fx[idx].copy()
        fy = self.fy[idx].copy()
        cx = self.cx[idx].copy()
        cy = self.cy[idx].copy()

        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }

        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }

        # is_train_synthetic = self.is_train and self.data_source != 'real'
        is_train_synthetic = self.is_train and self.is_aug

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        # todo: how to augment fx, fy, cx, cy?
        img_patch, keypoints_2d, keypoints_2d_crop_vis_mask, keypoints_3d_crop_auge, keypoints_3d_full_auge, \
        smpl_params, has_smpl_params, img_size, img_patch_cv, \
        center_x_auge, center_y_auge, cam_cx_auge, auge_scale, keypoints_2d_aug_orig, rotated_img, rotated_cvimg = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    self.flip_2d_keypoint_permutation,
                                                                                                                  self.flip_3d_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.do_augment, augm_config, fx, cam_cx=cx, cam_cy=cy,
                                                                                                    is_train_synthetic=is_train_synthetic)


        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['transf_kinect2holo'] = self.transf_kinect2holo[idx].copy()
        item['transf_holo2pv'] = self.transf_holo2pv[idx].copy()

        # if self.data_source != 'real':
        #     item['transf_kinect2holo'], item['transf_holo2pv'] = np.eye(4), np.eye(4)
        # else:
        #     seq_name = self.seq_names[idx]
        #     item['transf_kinect2holo'], item['transf_holo2pv'] = self.get_transf_matrices_per_frame(image_file, seq_name)

        item['img'] = img_patch
        item['imgname'] = image_file  # '/mnt/ssd/egobody_release/egocentric_color/recording_20220415_S36_S35_01/2022-04-15-161202/PV/132945055822281630_frame_02030.jpg'
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)  # [25, 3]
        item['keypoints_3d'] = keypoints_3d_crop_auge.astype(np.float32)  # [24, 3]
        item['keypoints_3d_full'] = keypoints_3d_full_auge.astype(np.float32)
        item['keypoints_2d_vis_mask'] = keypoints_2d_crop_vis_mask  # [25] vis mask for openpose joint in augmented cropped img
        item['img_size'] = 1.0 * img_size[::-1].copy()  # array([1080., 1920.])
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['idx'] = idx
        item['gender'] = gender

        item['fx'] = (fx / self.fx_norm_coeff).astype(np.float32)
        item['fy'] = (fy / self.fy_norm_coeff).astype(np.float32)
        item['cam_cx'] = cam_cx_auge.astype(np.float32)
        item['cam_cy'] = cy.astype(np.float32)

        item['vx'] = 2 * math.atan(img_size[1] / (2 * fx))


        # augmented
        item['orig_keypoints_2d'] = keypoints_2d_aug_orig.astype(np.float32)
        item['box_center'] = np.array([center_x_auge, center_y_auge]).astype(np.float32)
        item['box_size'] = (bbox_size * auge_scale).astype(np.float32)
        item['orig_img'] = rotated_img  # original img rotate around (center_x_auge, center_y_auge)


        # import pdb;  pdb.set_trace()
        return item
