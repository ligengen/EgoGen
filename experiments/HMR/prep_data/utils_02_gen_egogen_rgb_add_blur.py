# import open3d as o3d
import time
import numpy as np
import torch
import smplx
import os
import glob
import pdb
import copy
import cv2
# import PIL.Image as pil_img
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool
from blurgenerator import motion_blur
import datetime


#replace the data_root with your own path to the egogen rgb folder
data_root = ""

def add_noise(params):
    data_root, scene_name, frame_id = params
    image_path = os.path.join(data_root, scene_name, 'rgb', '{}.jpg'.format(frame_id))
    save_path = os.path.join(data_root, scene_name, 'rgb_blur', '{}.jpg'.format(frame_id))

    image = cv2.imread(image_path)
    
    psf = np.zeros((50, 50, 3))
    psf = cv2.ellipse(psf, 
                    (25, 25), # center
                    (22, 0), # axes -- 22 for blur length, 0 for thin PSF 
                    15, # angle of motion in degrees
                    0, 360, # ful ellipse, not an arc
                    (1, 1, 1), # white color
                    thickness=-1) # filled

    psf /= psf[:,:,0].sum() # normalize by sum of one channel 
                            # since channels are processed independently

    image = cv2.filter2D(image, -1, psf)

    cv2.imwrite("./", image)

    return 

# Add noise at probability 0.4
def add_noise_random(params):
    data_root, scene_name, frame_id = params
    image_path = os.path.join(data_root, scene_name, 'rgb', '{}.jpg'.format(frame_id))
    save_path = os.path.join(data_root, scene_name, 'rgb_blur', '{}.jpg'.format(frame_id))

    tmp = np.random.rand()
    if tmp > 0.4:
        os.system("cp %s %s"%(image_path, save_path))
        return

    image = cv2.imread(image_path)
    size = np.random.randint(20, 50)
    angle = np.random.randint(0, 360)
    image = motion_blur(image, size=size, angle=angle)
    cv2.imwrite(save_path, image)
    return



if __name__ == "__main__":

    scene_name_list = ['cab_e', 'cab_g_benches', 'cab_h_tables', 'cnb_dlab_0215', 'cnb_dlab_0225', 'foodlab_0312',
                       'kitchen_gfloor', 'seminar_d78', 'seminar_d78_0318', 'seminar_g110', 'seminar_g110_0315',
                       'seminar_g110_0415', 'seminar_h52', 'seminar_h53_0218', 'seminar_j716']


    for scene_name in scene_name_list:
        print(scene_name)
        print(datetime.datetime.now())
        save_dir = os.path.join(data_root, scene_name, 'rgb_blur')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        path = os.path.join(data_root, scene_name)
        num_frames = len(glob.glob1(path+"/rgb","*.jpg"))
        print("There are %d frames in scene %s"%(num_frames, scene_name))

        params = []
        for frame_id in range(1, num_frames+1):
            params.append((data_root, scene_name, frame_id))

        try:
            with Pool(20) as pool:
                pool.map(add_noise_random, params)
        except Exception as e:
            print(e)


