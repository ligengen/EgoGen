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
from .utils_scene import get_example
from ..utils.pose_utils import *
from ..utils.camera import create_camera
from .dataset_utils import get_right_full_img_pth, get_transf_matrices, parse_img_full_path
from ..utils.geometry import *

class ImageDatasetEgoBodyScene(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 split='train',
                 spacing=1,
                 add_scale=1.0,
                 device=None,
                 do_augment=False,
                 bps_type='joints',
                 bps_norm_type='type1',
                 scene_type='whole_scene',
                 scene_cube_normalize=False,
                 scene_cano=False,
                 scene_cano_norm=False,
                 data_gender='both',
                 scene_downsample_rate=1,
                 body_rep=None,
                 body_rep_normalize=False,
                 body_rep_normalize_sep=False,
                 # load_scene_normal=False,
                 load_stage1_transl=False,
                 stage1_result_path='',
                 scene_crop_by_stage1_transl=False,
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDatasetEgoBodyScene, self).__init__()
        self.train = train
        self.split = split
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.fx_norm_coeff = self.cfg.CAM.FX_NORM_COEFF
        self.fy_norm_coeff = self.cfg.CAM.FY_NORM_COEFF
        self.cx_norm_coeff = self.cfg.CAM.CX_NORM_COEFF
        self.cy_norm_coeff = self.cfg.CAM.CY_NORM_COEFF

        self.img_dir = img_dir
        self.data = np.load(dataset_file)
        with open(join(self.img_dir, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
            self.transf_matrices = pkl.load(fp)

        self.imgname = self.data['imgname']

        [self.imgname, self.seq_names, _] = zip(*[get_right_full_img_pth(x, self.img_dir) for x in self.imgname])   # absolute dir
        self.seq_names = [basename(x) for x in self.seq_names][::spacing]
        self.imgname = self.imgname[::spacing]

        body_permutation_2d = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]  # for openpose 25 topology
        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]  # for smpl 24 topology
        # extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        # flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        self.flip_2d_keypoint_permutation = body_permutation_2d
        self.flip_3d_keypoint_permutation = body_permutation_3d

        # num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center'][::spacing]
        self.scale = self.data['scale'][::spacing] * add_scale
        # self.scale = self.data['scale'].reshape(len(self.center), -1).max(axis=-1) / 200.0  # todo: ?

        self.has_smpl = np.ones(len(self.imgname))
        self.global_orient_kinect = self.data['global_orient'].astype(np.float)[::spacing]
        self.transl_kinect = self.data['transl'].astype(np.float)[::spacing]
        self.body_pose = self.data['pose'].astype(np.float)[::spacing]  # [n_sample, 69]
        self.betas = self.data['shape'].astype(np.float)[::spacing]
        # for k in ['global_orient', 'transl', 'pose', 'betas']:
        #     setattr(self, k, self.data[k].astype(np.float)[::spacing])
        # self.global_orient_pv = np.zeros(self.global_orient_kinect.shape)
        # self.transl_pv = np.zeros(self.transl_kinect.shape)
        self.global_orient_pv = self.data['global_orient_pv'].astype(np.float)[::spacing]  # [n_sample, 3]
        self.transl_pv = self.data['transl_pv'].astype(np.float)[::spacing]

        self.cx = self.data['cx'].astype(np.float)[::spacing]
        self.cy = self.data['cy'].astype(np.float)[::spacing]
        self.fx = self.data['fx'].astype(np.float)[::spacing]
        self.fy = self.data['fy'].astype(np.float)[::spacing]

        # self.cx = np.zeros(len(self.imgname))
        # self.cy = np.zeros(len(self.imgname))
        # self.fx = np.zeros(len(self.imgname))
        # self.fy = np.zeros(len(self.imgname))

        keypoints_openpose = self.data['valid_keypoints'][::spacing]
        self.keypoints_2d = keypoints_openpose
        # self.keypoints_3d_pv = np.zeros([len(self.imgname), 45, 3])  # smpl keypoints in pv frame coord
        self.keypoints_3d_pv = self.data['3d_joints_pv'].astype(np.float)[::spacing]

        # Get gender data, if available
        # try:
        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)

        self.load_stage1_transl = load_stage1_transl
        if self.load_stage1_transl:
            with open(stage1_result_path, 'rb') as fp:
                stage1_result = pkl.load(fp)
            self.stage1_transl_full = stage1_result['pred_cam_full_list'].astype(np.float)[::spacing]  # [n_samples, 3]?  todo

        ######## get mean/var for body_rep (to normalize for diffusion model)
        if body_rep is not None and body_rep_normalize and split=='train' and self.train:
            if body_rep == 'pose6d':
                global_orient_pv_all = torch.from_numpy(self.global_orient_pv).float()
                body_pose_all = torch.from_numpy(self.body_pose).float()
                full_pose_aa_all = torch.cat([global_orient_pv_all, body_pose_all], dim=1).reshape(-1, 24, 3)  # [n, 24, 3]
                full_pose_rotmat_all = aa_to_rotmat(full_pose_aa_all.reshape(-1, 3)).view(-1, 24, 3, 3)  # [bs, 24, 3, 3]
                full_pose_rot6d_all = rotmat_to_rot6d(full_pose_rotmat_all.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 24, 6).reshape(-1, 24*6)  # [n, 144]
                full_pose_rot6d_all = full_pose_rot6d_all.detach().cpu().numpy()
                Xmean = full_pose_rot6d_all.mean(axis=0)  # [d]
                Xstd = full_pose_rot6d_all.std(axis=0)  # [d]
                prefix = 'preprocess_stats'
                os.makedirs(prefix) if not os.path.exists(prefix) else None
                if not body_rep_normalize_sep:
                    Xstd[0:6] = Xstd[0:6].mean() / 1.0  # for global orientation
                    Xstd[6:] = Xstd[6:].mean() / 1.0  # for body pose
                    # full_pose_rot6d_all_norm = (full_pose_rot6d_all - Xmean) / Xstd
                    np.savez_compressed('preprocess_stats/{}_{}.npz'.format(prefix, body_rep), Xmean=Xmean, Xstd=Xstd)
                    print('[INFO] mean/std for body_rep [{}] saved.'.format(body_rep))
                else:
                    for j in range(24):
                        Xstd[(6*j):(6*(j+1))] = Xstd[(6*j):(6*(j+1))].mean() / 1.0
                    np.savez_compressed('preprocess_stats/{}_{}_body_rep_normalize_sep.npz'.format(prefix, body_rep), Xmean=Xmean, Xstd=Xstd)
                    print('[INFO] mean/std for body_rep [{}] saved. (body_rep_normalize_sep=True)'.format(body_rep))

            if body_rep == 'pose_aa':
                global_orient_pv_all = torch.from_numpy(self.global_orient_pv).float()
                body_pose_all = torch.from_numpy(self.body_pose).float()
                full_pose_aa_all = torch.cat([global_orient_pv_all, body_pose_all], dim=1).reshape(-1, 24, 3)  # [n, 24, 3]
                full_pose_aa_all = full_pose_aa_all.reshape(-1, 24*3).detach().cpu().numpy()  # [n, 72]
                Xmean = full_pose_aa_all.mean(axis=0)  # [d]
                Xstd = full_pose_aa_all.std(axis=0)  # [d]
                Xstd[0:3] = full_pose_aa_all[:, 0:3].std()  # for global orientation
                Xstd[3:] = full_pose_aa_all[:, 3:].std()  # for body pose
                prefix = 'preprocess_stats'
                os.makedirs(prefix) if not os.path.exists(prefix) else None
                np.savez_compressed('preprocess_stats/{}_{}.npz'.format(prefix, body_rep), Xmean=Xmean, Xstd=Xstd)
                print('[INFO] mean/std for body_rep [{}] saved.'.format(body_rep))


        ########### get mask
        if data_gender == 'female' or data_gender == 'male':
            if data_gender == 'female':
                mask = self.gender == 1
            elif data_gender == 'male':
                mask = self.gender == 0
            self.imgname = np.array(self.imgname)[mask]
            self.seq_names = np.array(self.seq_names)[mask]
            self.keypoints_2d = self.keypoints_2d[mask]
            self.keypoints_3d_pv = self.keypoints_3d_pv[mask]
            self.gender = self.gender[mask]
            self.center = self.center[mask]
            self.scale = self.scale[mask]
            self.has_smpl = self.has_smpl[mask]
            self.global_orient_kinect = self.global_orient_kinect[mask]
            self.transl_kinect = self.transl_kinect[mask]
            self.body_pose = self.body_pose[mask]
            self.betas = self.betas[mask]
            self.global_orient_pv = self.global_orient_pv[mask]
            self.transl_pv = self.transl_pv[mask]
            self.cx = self.cx[mask]
            self.cy = self.cy[mask]
            self.fx = self.fx[mask]
            self.fy = self.fy[mask]



        self.smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male')
        self.smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female')

        self.dataset_len = len(self.imgname)
        print('[INFO] find {} samples in {}.'.format(self.dataset_len, dataset_file))

        ########### read scene pcd
        self.scene_type = scene_type
        # self.load_whole_scene = load_whole_scene
        if self.scene_type == 'whole_scene':
            self.scene_cube_normalize = scene_cube_normalize
            # with open(os.path.join(self.img_dir, 'proHMR_scene_preprocess/pcd_verts_dict_{}.pkl'.format(split)), 'rb') as f:
            with open(os.path.join(self.img_dir, 'Egohmr_scene_preprocess_s1/pcd_verts_dict_{}.pkl'.format(split)), 'rb') as f:
                self.pcd_verts_dict_whole_scene = pkl.load(f)
            # with open(os.path.join(self.img_dir, 'proHMR_scene_preprocess/pcd_color_dict_{}.pkl'.format(split)), 'rb') as f:
            #     self.pcd_colors_dict_whole_scene = pkl.load(f)
            # with open(os.path.join(self.img_dir, 'proHMR_scene_preprocess/map_dict_{}.pkl'.format(split)), 'rb') as f:
            with open(os.path.join(self.img_dir, 'Egohmr_scene_preprocess_s1/map_dict_{}.pkl'.format(split)), 'rb') as f:
                self.pcd_map_dict_whole_scene = pkl.load(f)
        elif self.scene_type == 'cube_nowall':
            # self.pcd_root = os.path.join(self.img_dir, 'proHMR_scene_preprocess_cube_nowall')
            if not scene_crop_by_stage1_transl:
                self.pcd_root = os.path.join(self.img_dir, 'Egohmr_scene_preprocess_cube_s2_from_gt')  # train
            else:
                self.pcd_root = os.path.join(self.img_dir, 'Egohmr_scene_preprocess_cube_s2_from_pred')  # test
            self.scene_cube_normalize = scene_cube_normalize
        elif self.scene_type == 'cube_nowall_2w':
            if not scene_crop_by_stage1_transl:
                self.pcd_root = os.path.join(self.img_dir, 'proHMR_scene_preprocess_cube_nowall_2w')
            else:
                self.pcd_root = os.path.join(self.img_dir, 'proHMR_scene_preprocess_cube_nowall_2w_for_eval')
            self.scene_cube_normalize = scene_cube_normalize
        # elif self.scene_type == 'cube_withwall':
        #     self.pcd_root = os.path.join(self.img_dir, 'proHMR_scene_preprocess_cube_withwall')
        #     self.scene_cube_normalize = scene_cube_normalize
        else:
            print('[ERROR] wrong scene_type!')
            exit()

        # self.load_scene_normal = load_scene_normal
        # if load_scene_normal:
        #     if not scene_crop_by_stage1_transl:
        #         self.scene_normal_root = os.path.join(self.img_dir, 'proHMR_scene_preprocess_cube_nowall_2w_normals_pv')
        #     else:
        #         self.scene_normal_root = os.path.join(self.img_dir, 'proHMR_scene_preprocess_cube_nowall_2w_normals_pv_for_eval')




        df = pd.read_csv(os.path.join(self.img_dir, 'data_info_release.csv'))
        recording_name_list = list(df['recording_name'])
        scene_name_list = list(df['scene_name'])
        self.scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        self.add_trans = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.scene_cano = scene_cano
        self.scene_cano_norm = scene_cano_norm
        self.scene_downsample_rate = scene_downsample_rate
        # min_x_list, max_x_list, min_y_list, max_y_list, min_z_list, max_z_list = [], [], [], [], [], []
        # for key in self.pcd_verts_dict.keys():
        #     cur_verts = self.pcd_verts_dict[key]
        #     min_x = cur_verts[:, 0].min()
        #     max_x = cur_verts[:, 0].max()
        #     min_y = cur_verts[:, 1].min()
        #     max_y = cur_verts[:, 1].max()
        #     min_z = cur_verts[:, 2].min()
        #     max_z = cur_verts[:, 2].max()
        #     min_x_list.append(min_x)
        #     max_x_list.append(max_x)
        #     min_y_list.append(min_y)
        #     max_y_list.append(max_y)
        #     min_z_list.append(min_z)
        #     max_z_list.append(max_z)
        # print(1)  # train: x: -4.5-4.9, y: -2.7-1.5, z: -2.0-7.5, val: x: -3.9-4.9, y: -2.4-1.5, z: -0.9-7.1

        if self.scene_type == 'whole_scene':
            bps_folder_name = 'body_bps_encoding'
        elif self.scene_type == 'cube_nowall':
            bps_folder_name = 'body_bps_encoding_cube_nowall'
        elif self.scene_type == 'cube_nowall_2w':
            bps_folder_name = None  # todo
        elif self.scene_type == 'cube_withwall':
            bps_folder_name = 'body_bps_encoding_cube_withwall'
        if not self.scene_type == 'cube_nowall_2w':
            with open(os.path.join(self.img_dir, bps_folder_name, 'body_{}_bps_{}.pkl'.format(bps_type, split)), 'rb') as f:
                self.bps_encoding_dict = pkl.load(f)
            self.bps_norm_type = bps_norm_type
            self.bps_mean = 0.46
            self.bps_std = 0.34


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
        image_file = join(self.img_dir, self.imgname[idx])  # absolute path
        seq_name = self.seq_names[idx]
        keypoints_2d = self.keypoints_2d[idx].copy()  # [25, 3], openpose joints
        # keypoints_3d = self.keypoints_3d[idx].copy()
        keypoints_3d = self.keypoints_3d_pv[idx][0:24].copy()  # [24, 3], smpl joints

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


        item = {}
        item['transf_kinect2holo'], item['transf_holo2pv'] = self.get_transf_matrices_per_frame(image_file, seq_name)

        pcd_trans_kinect2pv = np.matmul(item['transf_holo2pv'], item['transf_kinect2holo'])
        pcd_trans_kinect2pv = np.matmul(self.add_trans, pcd_trans_kinect2pv)

        temp = "/".join(image_file.split('/')[-5:])
        if self.scene_type == 'whole_scene':
            scene_pcd_verts = self.pcd_verts_dict_whole_scene[self.pcd_map_dict_whole_scene[temp]]  # [20000, 3], in kinect main coord
            # scene_pcd_colors = self.pcd_colors_dict[self.pcd_map_dict[temp]]  # [20000, 3]
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_kinect2pv)
        elif self.scene_type == 'cube_nowall' or self.scene_type == 'cube_nowall_2w' or self.scene_type == 'cube_withwall':
            recording_name = image_file.split('/')[-4]
            img_name = image_file.split('/')[-1]
            scene_pcd_path = os.path.join(self.pcd_root, self.split, recording_name, image_file.split('/')[-3], img_name[:-3]+'npy')
            scene_pcd_verts = np.load(scene_pcd_path)  # in scene coord
            # transformation from master kinect RGB camera to scene mesh
            calib_trans_dir = os.path.join(self.img_dir, 'calibrations', recording_name)
            cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')
            with open(os.path.join(cam2world_dir, self.scene_name_dict[recording_name] + '.json'), 'r') as f:
                trans_scene_to_main = np.array(json.load(f)['trans'])
            trans_scene_to_main = np.linalg.inv(trans_scene_to_main)
            pcd_trans_scene2pv = np.matmul(pcd_trans_kinect2pv, trans_scene_to_main)
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_scene2pv)  # nowall: 5000, withwall: 5000+30*30*5=9500

        # if self.load_scene_normal:
        #     recording_name = image_file.split('/')[-4]
        #     img_name = image_file.split('/')[-1]
        #     path = os.path.join(self.scene_normal_root, self.split, recording_name, image_file.split('/')[-3], img_name[:-3]+'npy')
        #     scene_normals_pv = np.load(path)   # in pv coord

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        # todo: how to augment fx, fy, cx, cy?
        img_patch, keypoints_2d, keypoints_2d_crop_vis_mask, keypoints_3d_crop_auge, keypoints_3d_full_auge, \
        smpl_params, has_smpl_params, img_size, img_patch_cv, \
        center_x_auge, center_y_auge, cam_cx_auge, auge_scale, keypoints_2d_aug_orig, rotated_img, rotated_cvimg, scene_pcd_verts_full_auge = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    self.flip_2d_keypoint_permutation,
                                                                                                                  self.flip_3d_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.do_augment, augm_config, fx, cam_cx=cx, cam_cy=cy,
                                                                                                                                                          scene_pcd_verts=scene_pcd_verts, scene_pcd_colors=None,
                                                                                                                                                          smpl_male=self.smpl_male, smpl_female=self.smpl_female, gender=gender)

        # ######################################################### debug: vis
        # # camera_center = np.array([cx, cy])
        # camera_pose = np.eye(4)
        # camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        # light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        # base_color = (1.0, 193 / 255, 193 / 255, 1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=base_color
        # )
        #
        # ############# vis augemented 2d openpose joints on auge crop img
        # img_patch_cv = pil_img.fromarray((img_patch_cv).astype(np.uint8))
        # draw = ImageDraw.Draw(img_patch_cv)
        # keypoints_2d_vis = keypoints_2d.copy()
        # keypoints_2d_vis[:, :-1] = (keypoints_2d[:, :-1] + 0.5) * 224
        # for k in range(25):
        #     draw.ellipse((keypoints_2d_vis[k][0] - 4, keypoints_2d_vis[k][1] - 4,
        #                   keypoints_2d_vis[k][0] + 4, keypoints_2d_vis[k][1] + 4), fill=(0, 255, 0, 0))
        # img_patch_cv.show()
        #
        # ############# project augemented smpl params on original img
        # if self.gender[idx] == 0:
        #     smpl_model = self.smpl_male
        # else:
        #     smpl_model = self.smpl_female
        # body_params_dict_new = {}
        # body_params_dict_new['global_orient'] = smpl_params['global_orient']
        # body_params_dict_new['transl'] = smpl_params['transl']
        # body_params_dict_new['body_pose'] = smpl_params['body_pose']
        # body_params_dict_new['betas'] = smpl_params['betas']
        # for key in body_params_dict_new.keys():
        #     body_params_dict_new[key] = torch.FloatTensor(body_params_dict_new[key]).unsqueeze(0)
        # cur_vertices_full_auge = smpl_model(**body_params_dict_new).vertices.detach().cpu().numpy().squeeze()
        #
        # camera_full = pyrender.camera.IntrinsicsCamera(
        #     fx=fx, fy=fy,
        #     cx=cam_cx_auge, cy=cy)
        # # cur_vertices_full_auge = smpl_model(**body_params_dict_new).vertices.detach().cpu().numpy().squeeze()
        # # cur_vertices_full_auge: same procedure as did to keypoints
        # body = trimesh.Trimesh(cur_vertices_full_auge, self.smpl_male.faces, process=False)
        # body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
        #
        # # input_img = cv2.imread(self.imgname[idx])[:, :, ::-1]
        # scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
        #                        ambient_light=(0.3, 0.3, 0.3))
        # scene.add(camera_full, pose=camera_pose)
        # scene.add(light, pose=camera_pose)
        # scene.add(body_mesh, 'mesh')
        # r = pyrender.OffscreenRenderer(viewport_width=1920,
        #                                viewport_height=1080,
        #                                point_size=1.0)
        # color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        #
        # color = color.astype(np.float32) / 255.0
        # alpha = 1.0  # set transparency in [0.0, 1.0]
        # color[:, :, -1] = color[:, :, -1] * alpha
        # color = pil_img.fromarray((color * 255).astype(np.uint8))
        # output_img = pil_img.fromarray((rotated_cvimg).astype(np.uint8))  # on original img?
        # output_img.paste(color, (0, 0), color)
        # output_img.show()
        # print(1)
        #
        # ############### project augemented 3d smpl joints on original img
        # camera_center_holo = torch.tensor([cam_cx_auge, cy]).view(-1, 2)
        # camera_holo_kp = create_camera(camera_type='persp_holo',
        #                                focal_length_x=torch.tensor([fx]).unsqueeze(0),
        #                                focal_length_y=torch.tensor([fy]).unsqueeze(0),
        #                                center=camera_center_holo,
        #                                batch_size=1)
        #
        # joints = torch.from_numpy(keypoints_3d_full_auge).float().unsqueeze(0)  # [1, n_joints, 3]
        # gt_joints_2d = camera_holo_kp(joints)  # project 2d joints on holo images of gt body [1, n_joints, 2]
        # gt_joints_2d = gt_joints_2d.squeeze().detach().cpu().numpy()  # [n_joints, 2]
        # draw = ImageDraw.Draw(output_img)
        # for k in range(len(gt_joints_2d)):
        #     draw.ellipse((gt_joints_2d[k][0] - 4, gt_joints_2d[k][1] - 4,
        #                   gt_joints_2d[k][0] + 4, gt_joints_2d[k][1] + 4), fill=(255, 0, 0, 0))
        #
        # line_joint_indexs = [[0, 1], [0, 2], [0,3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10],
        #                      [8,11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
        #                      [18, 20], [19, 21], [20, 22], [21, 23]]
        # # drawing line
        # for index_pair in line_joint_indexs:
        #     draw.line(xy=[gt_joints_2d[index_pair[0]][0],
        #                   gt_joints_2d[index_pair[0]][1],
        #                   gt_joints_2d[index_pair[1]][0],
        #                   gt_joints_2d[index_pair[1]][1]],
        #               fill=(255, 0, 0, 0), width=1)
        #
        # output_img.show()
        # print(1)


        # These are the keypoints in the original image coordinates (before cropping)
        # orig_keypoints_2d = self.keypoints_2d[idx].copy()


        item['img'] = img_patch
        item['imgname'] = image_file  # '/mnt/ssd/egobody_release/egocentric_color/recording_20220415_S36_S35_01/2022-04-15-161202/PV/132945055822281630_frame_02030.jpg'
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)  # [25, 3]
        item['keypoints_3d'] = keypoints_3d_crop_auge.astype(np.float32)  # [24, 3]
        item['keypoints_3d_full'] = keypoints_3d_full_auge.astype(np.float32)
        item['keypoints_2d_vis_mask'] = keypoints_2d_crop_vis_mask  # [25] vis mask for openpose joint in augmented cropped img
        # item['orig_keypoints_2d'] = orig_keypoints_2d.astype(np.float32)
        # item['box_center'] = self.center[idx].copy().astype(np.float32)
        # item['box_size'] = bbox_size.astype(np.float32)
        item['img_size'] = 1.0 * img_size[::-1].copy()  # array([1080., 1920.])
        item['smpl_params'] = smpl_params
        for key in item['smpl_params'].keys():
            item['smpl_params'][key] = item['smpl_params'][key].astype(np.float32)
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['idx'] = idx
        item['gender'] = gender

        item['fx'] = (fx / self.fx_norm_coeff).astype(np.float32)
        item['fy'] = (fy / self.fy_norm_coeff).astype(np.float32)
        # item['cam_cx'] = (cx / self.cx_norm_coeff).astype(np.float32)
        # item['cam_cx'] = (cam_cx_auge / self.cx_norm_coeff).astype(np.float32)
        # item['cam_cy'] = (cy / self.cy_norm_coeff).astype(np.float32)
        item['cam_cx'] = cam_cx_auge.astype(np.float32)
        item['cam_cy'] = cy.astype(np.float32)

        item['vx'] = 2 * math.atan(img_size[1] / (2 * fx))


        # augmented
        item['orig_keypoints_2d'] = keypoints_2d_aug_orig.astype(np.float32)
        item['box_center'] = np.array([center_x_auge, center_y_auge]).astype(np.float32)
        item['box_size'] = (bbox_size * auge_scale).astype(np.float32)
        item['orig_img'] = rotated_img  # original img rotate around (center_x_auge, center_y_auge)

        # scene_pcd_verts_full_auge = np.transpose(scene_pcd_verts_full_auge, (1, 0)).astype(np.float32)
        scene_pcd_verts_full_auge = scene_pcd_verts_full_auge.astype(np.float32)  # [n_pts, 3]
        if self.scene_type == 'whole_scene':
            if self.scene_cube_normalize and (not self.scene_cano):
                scene_pcd_verts_full_auge = scene_pcd_verts_full_auge / 5.0  # todo: how to normalize?
        elif self.scene_type == 'cube_nowall' or self.scene_type == 'cube_nowall_2w' or self.scene_type == 'cube_withwall':
            if self.scene_cube_normalize and (not self.scene_cano):  # if canonicalize scene, normlize later
                scene_pcd_verts_full_auge = scene_pcd_verts_full_auge / 4.0
                # scene_pcd_verts_full_auge[:, -1] = scene_pcd_verts_full_auge[:, -1] - 2  # todo: how to normalize? transl z by -2
        scene_pcd_verts_full_auge = scene_pcd_verts_full_auge[::self.scene_downsample_rate]
        item['scene_pcd_verts_full'] = scene_pcd_verts_full_auge  # [20000, 3]

        # if self.load_scene_normal:
        #     item['scene_normals'] = scene_normals_pv.astype(np.float32)
        # else:
        #     item['scene_normals'] = 0.0

        if self.load_stage1_transl:
            item['stage1_transl_full'] = self.stage1_transl_full[idx].astype(np.float32)


            # ############### todo: new BPS
        # if not self.scene_type == 'cube_nowall_2w':
        #     bps_encoding = self.bps_encoding_dict[temp]  # [24/66/6890]
        #     # todo: how to normalize?
        #     if self.bps_norm_type == 'type1':
        #         bps_encoding = bps_encoding.astype(np.float32) / 2.0
        #     elif self.bps_norm_type == 'type2':
        #         bps_encoding = bps_encoding.astype(np.float32) - 1.0
        #     elif self.bps_norm_type == 'type3':
        #         bps_encoding = (bps_encoding.astype(np.float32) - self.bps_mean) / self.bps_std
        #     elif self.bps_norm_type == 'none':
        #         bps_encoding = bps_encoding.astype(np.float32)
        # else:
        #     bps_encoding = 0  # todo
        # item['bps_encoding'] = bps_encoding


        # import pdb;  pdb.set_trace()
        return item
