import cv2
import time
import open3d as o3d
import numpy as np
import torch
import smplx
import os
import glob
import copy
import PIL.Image as pil_img
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation as R
from kinect_depth_noise.process_kinect_depth import add_kinect_noise_to_depth
from multiprocessing import Pool

# replace this with the path to the egobody depth folder
data_root = ""

parser = argparse.ArgumentParser(description='ProHMR training code')
parser.add_argument('--egobody_depth_path',
                    type=str,
                    default=None,
                    help='Path to egogen clean depth data')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Number of workers for data loading')
args = parser.parse_args()

if args.egobody_depth_path is None:
    args.egobody_depth_path = data_root


def add_noise(params):
    scene_name, frame_id = params
    print('processing ', scene_name, ' ', frame_id)
    depth_clean = np.load(os.path.join(args.data_root, scene_name, 'depth_clean', '{}.npy'.format(frame_id)))
    depth_noisy, _ = add_kinect_noise_to_depth(depth_clean, dot_pattern,
                                               scale_factor=100,
                                               baseline_m=0.03,
                                               std=0.05,
                                               size_filt=6,
                                               focal_length=200,
                                               a_min=0.01,
                                               a_max=5,
                                               w=320,
                                               h=288)
    depth_noisy[depth_noisy==float('-inf')] = float('inf')
    depth_noisy[depth_noisy==float('inf')] = 0
    depth_noisy[depth_clean==0] = 0

    depth_noisy[depth_noisy>=5] = 0
    depth_noisy[depth_noisy <=0.01] = 0
    depth = depth_noisy * 1000 * 8
    depth = depth.astype(np.uint16)
    depth = pil_img.fromarray(depth)
    # depth.show()
    depth_save_dir = os.path.join(args.data_root, scene_name, 'depth_noisy')
    depth.save(os.path.join(depth_save_dir, '{}.png').format(frame_id))


if __name__=='__main__':

    scene_name_list = ['cab_e', 'cab_g_benches', 'cab_h_tables', 'cnb_dlab_0215', 'cnb_dlab_0225', 'foodlab_0312',
                       'kitchen_gfloor', 'seminar_d78', 'seminar_d78_0318', 'seminar_g110', 'seminar_g110_0315',
                       'seminar_g110_0415', 'seminar_h52', 'seminar_h53_0218', 'seminar_j716']
    dot_pattern = cv2.imread("kinect_depth_noise/kinect-pattern_3x3.png", 0)

    for scene_name in scene_name_list:
        print(scene_name)

        depth_save_dir = os.path.join(args.data_root, scene_name, 'depth_noisy')
        if not os.path.exists(depth_save_dir):
            os.makedirs(depth_save_dir)


        params = []
        for frame_id in range(1, 7001):
            params.append((scene_name, frame_id))

        try:
            with Pool(16) as pool:
                pool.map(add_noise, params)
        except Exception as e:
            print(e)

