from genericpath import isdir
import os
from os.path import join, basename, exists, isdir
from glob import glob
from re import I

import pickle as pkl

import numpy as np
import glob
import ast
import json
from tqdm import tqdm
# import open3d as o3d
import sys
sys.path.append(sys.path[0] + '/..')

import socket
hostname = socket.gethostname()

ETHPV_DATA_ROOT_CLUSTER = '/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/szhang/egocaptures'
ETHPV_DATA_ROOT_ATLAS = '/vlg-nfs/szhang/egocaptures'
ETHPV_DATA_ROOT_LOCAL = '/run/user/1001/gvfs/smb-share:server=tang.scratch.inf.ethz.ch,share=scratch-tang/cluster/szhang/egocaptures'

# TRANSF_MTX_SAVE_DIR = 'egocapture_transf_mtxs'

if hostname=='vlg-atlas.inf.ethz.ch':
    ETHFV_ROOT = ETHPV_DATA_ROOT_ATLAS
    TRANSF_MTX_SAVE_DIR = '/vlg-nfs/shared/egocapture_transf_mtxs'
elif hostname == 'vlg01':
    ETHFV_ROOT = ETHPV_DATA_ROOT_LOCAL
    TRANSF_MTX_SAVE_DIR = '/home/qianlima/mnt/egocapture_transf_mtxs'
else:
    ETHFV_ROOT = ETHPV_DATA_ROOT_CLUSTER
    TRANSF_MTX_SAVE_DIR = '/cluster/scratch/qianlima/egocapture_transf_mtxs'


def parse_img_full_path(img_full_path):
    '''
    given image full path, e.g. '/vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_02_moh_lam/2021-09-11-150137/PV/132758390080109913_frame_03611.jpg'
    return the name of the session, sequence, and the image basename
    '''
    path_splitted = img_full_path.split('/')
    img_basename = path_splitted[-1]  # '132754997786014666_frame_01442.jpg'
    session = path_splitted[-5]
    seq = path_splitted[-4]
    fpv_recording_name = path_splitted[-3]  # '2021-09-07-164904'

    return session, seq, fpv_recording_name, img_basename

# def get_right_full_img_pth(imgname_in_npz, dataset):
#     '''
#     given a single image name packed in the npz, 
#     e.g. /local/home/zhqian/sp/data/ethfv/image/recording_20210911_s1_01_moh_lam/132758379560268230_frame_01888.jpg
#     get its correct full path stored in the vlg-atlas or cluster

#     returns:
#     record_folder_path: e.g. /vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam
#     img_pth_final: e.g. /vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam/<date-info>/PV/<timestamp>_frame_<kinet_frameid>.jpg
#     fpv_recording_name: e.g. 2021-09-18-155011
#     '''
#     img_dir = config.DATASET_FOLDERS[dataset]
#     # original img full path are like 
#     # `/vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam/2021-09-11-144522/PV`
#     datapth_prefix_old = '/local/home/zhqian/sp/data/ethfv/image/'
#     datapth_prefix_new = img_dir
#     record_folder_name = imgname_in_npz.replace(datapth_prefix_old, '').split('/')[0] # e.g. recording_20210911_s1_01_moh_lam
#     record_id = record_folder_name.rsplit('_', 4)[0] # e.g. recording_20210911
#     record_date = record_id.split('_')[1] # e.g. 20210911
#     record_id_short = 'record_{}'.format(record_date)

#     record_folder_path = join(img_dir, record_id_short, record_folder_name)
#     fpv_recording_name = [x for x in os.listdir(record_folder_path) if x.startswith('20')][0]
#     img_folder_final = join(record_folder_path, fpv_recording_name, 'PV')

#     img_pth_final = join(img_folder_final, basename(imgname_in_npz))
    
#     return img_pth_final, record_folder_path, fpv_recording_name

def get_right_full_img_pth(imgname_in_npz, img_dir):
    '''
    given a single image name packed in the npz, 
    e.g. hololens_data/record_20210911/recording_20210911_s1_01_moh_lam/2021-09-11-144522/PV/132758379563600210_frame_01898.jpg
    get its correct full path stored in the vlg-atlas or cluster

    returns:
    record_folder_path: e.g. /vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam
    img_pth_final: e.g. /vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam/<date-info>/PV/<timestamp>_frame_<kinet_frameid>.jpg
    fpv_recording_name: e.g. 2021-09-18-155011
    '''
    # import config
    # img_dir = config.DATASET_FOLDERS[dataset]
    # original img full path are like 
    # `/vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam/2021-09-11-144522/PV`
    # datapth_prefix_old = 'hololens_data'
    # print(imgname_in_npz, img_dir)
    session, seq, fpv_recording_name, img_basename = parse_img_full_path(imgname_in_npz)
    # img_pth_final = imgname_in_npz.replace(datapth_prefix_old, img_dir)
    try:
        imgname_in_npz = imgname_in_npz.split("egobody_release/")[1]
    except:
        pass
    img_pth_final = join(img_dir, imgname_in_npz)
    record_folder_path = join(img_dir, session, seq)
    # import pdb; pdb.set_trace()
    return img_pth_final, record_folder_path, fpv_recording_name

def get_transf_matrices(record_folder_path, fpv_recording_name, write_to_file=False):
    '''
    works per sequence

    record_folder_path: e.g. /vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam
    fpv_recording_name: e.g. 2021-09-18-155011
    write_to_file: if true, will write the acquired transf matrices to a npz file (one filel per sequence) (recommended)
                   if false, will this func can be used in pytorch dataloader and compute the transf matrices on-the-fly when loading the data (can possibly be slow)
    '''

    session, seq = record_folder_path.split('/')[-2], record_folder_path.split('/')[-1]

    holo2kinect_fn = join(record_folder_path, 'cal_trans/holo_to_kinect12.json')
    if not exists(holo2kinect_fn):
        print('{} does not have kinect to hololense calibration!'.format(record_folder_path))
        with open(join(TRANSF_MTX_SAVE_DIR, 'missing_kinect2holo_calib.txt'), 'a+') as fp:
            fp.write('{}\t{}'.format(session, seq))
        return
        
    with open(holo2kinect_fn, 'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

    pv_info_path = join(record_folder_path, fpv_recording_name, '{}_pv.txt'.format(fpv_recording_name))
    with open(pv_info_path) as f:
        lines = f.readlines()
    cx, cy, w, h = ast.literal_eval(lines[0])  # hololens pv camera infomation. cx, cy: camera center

    pv_fx_dict = {}
    pv_fy_dict = {}
    world2pv_transform_dict = {}
    for i, frame in enumerate(lines[1:]):
        frame = frame.split((','))
        cur_timestamp = int(frame[0])
        cur_fx = float(frame[1])
        cur_fy = float(frame[2])
        cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))
        cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

        pv_fx_dict[cur_timestamp] = cur_fx
        pv_fy_dict[cur_timestamp] = cur_fy
        world2pv_transform_dict[str(cur_timestamp)] = cur_world2pv_transform
    
    data_dict = {
        'trans_kinect2holo': trans_kinect2holo,
        'trans_world2pv': world2pv_transform_dict
        }

    if write_to_file:
        save_dir = join(TRANSF_MTX_SAVE_DIR, 'per_seq')
        os.makedirs(save_dir, exist_ok=True)
        save_fn = join(save_dir, '{}_transf_matrices.pkl'.format(seq))
        data_dict = {
                    'trans_kinect2holo': trans_kinect2holo,
                    'trans_world2pv': world2pv_transform_dict
        }
        with open(save_fn, 'wb') as fp:
            pkl.dump(data_dict, fp)

    else: 
        return data_dict
        
def get_holo_cam_intrinsics(record_folder_path, fpv_recording_name, write_to_file=False):
    '''
    works per sequence

    record_folder_path: e.g. /vlg-nfs/szhang/egocaptures/record_20210911/recording_20210911_s1_01_moh_lam
    fpv_recording_name: e.g. 2021-09-18-155011
    write_to_file: if true, will write the acquired transf matrices to a npz file (one filel per sequence) (recommended)
                   if false, will this func can be used in pytorch dataloader and compute the transf matrices on-the-fly when loading the data (can possibly be slow)
    '''

    session, seq = record_folder_path.split('/')[-2], record_folder_path.split('/')[-1]

    holo2kinect_fn = join(record_folder_path, 'cal_trans/holo_to_kinect12.json')
    if not exists(holo2kinect_fn):
        print('{} does not have kinect to hololense calibration!'.format(record_folder_path))
        with open(join(TRANSF_MTX_SAVE_DIR, 'missing_kinect2holo_calib.txt'), 'a+') as fp:
            fp.write('{}\t{}'.format(session, seq))
        
    pv_info_path = join(record_folder_path, fpv_recording_name, '{}_pv.txt'.format(fpv_recording_name))
    with open(pv_info_path) as f:
        lines = f.readlines()
    cx, cy, w, h = ast.literal_eval(lines[0])  # hololens pv camera infomation. cx, cy: camera center

    pv_fx_dict = {}
    pv_fy_dict = {}
    fl_dict = {}
    for i, frame in enumerate(lines[1:]):
        frame = frame.split((','))
        cur_timestamp = int(frame[0])
        cur_fx = float(frame[1])
        cur_fy = float(frame[2])
        cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))
        cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

        pv_fx_dict[cur_timestamp] = cur_fx
        pv_fy_dict[cur_timestamp] = cur_fy
        fl_dict[str(cur_timestamp)] = np.array([cur_fx, cur_fy])
    
    data_dict = {
        'principle_points': np.array([cx, cy]),
        'img_resl': np.array([w, h]),
        'focal_lengths': fl_dict
        }

    if write_to_file:
        save_dir = join(TRANSF_MTX_SAVE_DIR, 'per_seq')
        os.makedirs(save_dir, exist_ok=True)
        save_fn = join(save_dir, '{}_holo_intrinsics.pkl'.format(seq))
        import ipdb; ipdb.set_trace()
        with open(save_fn, 'wb') as fp:
            pkl.dump(data_dict, fp)

    else: 
        return data_dict
    
if __name__ == '__main__':
    import os
    import sys
    
    mode = sys.argv[1]
    sessions = sorted([x for x in os.listdir(ETHFV_ROOT) if (x.startswith('record') and isdir(join(ETHFV_ROOT, x)))])

    merge_all = bool(int(sys.argv[2]))
    if not merge_all:
        for session in sessions:
            session_dir = join(ETHFV_ROOT, session)
            sequences = [x for x in os.listdir(session_dir) if (x.startswith('recording') and isdir(join(session_dir, x)))]
            print('total: {} seqs!'.format(len(sequences)))
            for seq in tqdm(sequences):

                seq_dir = join(ETHFV_ROOT, session, seq)
                fpv_recording_name = [x for x in os.listdir(seq_dir) if x.startswith('20')][0]
                
                if mode == 'transf':
                    get_transf_matrices(seq_dir, fpv_recording_name, write_to_file=True)
                else:
                    get_holo_cam_intrinsics(seq_dir, fpv_recording_name, write_to_file=True)

    if merge_all:
        name = 'transf_matrices' if mode == 'transf' else 'holo_intrinsics'
        pklfiles = sorted(glob.glob(join(TRANSF_MTX_SAVE_DIR, 'per_seq', '*_{}.pkl'.format(name))))
        big_dict = {}
        for fn in pklfiles:
            with open(fn,'rb') as fp:
                data = pkl.load(fp)
            bn = basename(fn).replace('_{}.pkl'.format(name),'')
            big_dict[bn] = data
        import pickle as pkl

        if mode == 'transf':    
            with open(join(TRANSF_MTX_SAVE_DIR, 'transf_matrices_all_seqs.pkl'),'wb') as fp:
                pkl.dump(big_dict, fp)

        with open(join(TRANSF_MTX_SAVE_DIR, 'holo_intrinsics_all_seqs.pkl'),'wb') as fp:
            pkl.dump(big_dict, fp)
