import os
import numpy as np
import torch
from tqdm import tqdm
import copy
from scipy.spatial.transform import Rotation as R
import smplx
from os.path import join, exists, basename
import pickle as pkl
import time

# replace this with the path to the egobody folder
egobody_release_path = ""

def update_globalRT_for_smpl(body_param_dict, smpl_model, trans_to_target_origin, device, delta_T=None):
    '''
    input:
        body_params: array, [72], under camera coordinate
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''
    ### step (1) compute the shift of pelvis from the origin
    body_param_dict_torch = {}
    for key in body_param_dict.keys():
        body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key]).to(device)

    if delta_T is None:
        body_param_dict_torch['transl'] = torch.zeros([1, 3], dtype=torch.float32).to(device)
        body_param_dict_torch['global_orient'] = torch.zeros([1, 3], dtype=torch.float32).to(device)
        smpl_out = smpl_model(**body_param_dict_torch)
        delta_T = smpl_out.joints[0,0,:] # (3,)
        delta_T = delta_T.detach().cpu().numpy()
        joints_3d_kinect = smpl_out.joints[:, 0:24].detach().cpu().numpy() # [1, 24, 3]

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_param_dict['global_orient'][0]
    body_R_mat = R.from_rotvec(body_R_angle).as_matrix() # to a [3,3] rotation mat
    body_T = body_param_dict['transl'][0]
    body_mat = np.eye(4)
    body_mat[:-1,:-1] = body_R_mat
    body_mat[:-1, -1] = body_T + delta_T

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_param_dict)
    body_mat_new = np.dot(trans_to_target_origin, body_mat)
    body_R_new = R.from_matrix(body_mat_new[:-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(1,3)
    return body_params_dict_new

def get_transf_matrices_per_frame(img_name, seq_name, img_dir):
    with open(join(img_dir, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        transf_matrices = pkl.load(fp)
    transf_mtx_seq = transf_matrices[seq_name]
    kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
    holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix

    timestamp = basename(img_name).split('_')[0]
    holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
    return kinect2holo, holo2pv

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

    session, seq, fpv_recording_name, img_basename = parse_img_full_path(imgname_in_npz)
    img_pth_final = join(img_dir, imgname_in_npz)
    record_folder_path = join(img_dir, session, seq)
    return img_pth_final, record_folder_path, fpv_recording_name

def preprocess(split, egobody_release_path=None):

    prefix = ""
    # prefix = "/mnt/atlas_root"

    smpl_path = prefix + egobody_release_path +"smpl_spin_npz/"
    smplx_path = prefix + egobody_release_path + "smplx_spin_npz/"
    save_path = "../data/smplx_spin_npz_new/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    smpl_file_name = 'egocapture_%s_smpl.npz'
    smplx_file_name = 'egocapture_%s_smplx.npz'
    save_name = 'egocapture_%s_smplx.npz'

    device = torch.device('cuda')
    # device = torch.device('cpu')

    #read npz file
    smpl_file = np.load(os.path.join(smpl_path, smpl_file_name % split))
    smplx_file = np.load(os.path.join(smplx_path, smplx_file_name % split))

    #copy npz file smplx_file to save_file
    save_file = {}
    for key in smplx_file.keys():
        save_file[key] = smplx_file[key].copy()
        
    img_dir = prefix + egobody_release_path
    [save_file["imgname"], save_file["seq_names"], _] = zip(*[get_right_full_img_pth(x, img_dir) for x in save_file["imgname"]])
    save_file["seq_names"] = [basename(x) for x in save_file["seq_names"]][::1]
    
    save_file['fx'] = smpl_file['fx']
    save_file['fy'] = smpl_file['fy']
    save_file['cx'] = smpl_file['cx']
    save_file['cy'] = smpl_file['cy']

    
    save_file['global_orient_pv'] = np.zeros([50, 3])
    save_file['transl_pv'] = np.zeros([50, 3])
    save_file['3d_joints_pv'] = np.zeros([50, 127, 3])

    add_trans = np.array([[1.0, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    
    smplx_male_model = smplx.create(prefix + '../data/smplx_model', model_type='smplx', gender='male').to(device)
    smplx_female_model = smplx.create(prefix + '../data/smplx_model', model_type='smplx', gender='female').to(device)
    
    save_file['3d_joints_pv'] = np.zeros([len(save_file["imgname"]), 127, 3])

    for i in tqdm(range(0, len(save_file["imgname"]))):
        body_param_dict = {}
        body_param_dict['global_orient'] = save_file['global_orient'][i:(i+1)]
        body_param_dict['transl'] = save_file['transl'][i:(i + 1)]
        body_param_dict['body_pose'] = save_file['body_pose'][i:(i + 1)]
        body_param_dict['betas'] = save_file['betas'][i:(i + 1)]

        transf_kinect2holo, transf_holo2pv = get_transf_matrices_per_frame(save_file['imgname'][i], save_file['seq_names'][i], img_dir)
        trans_to_target_origin = np.matmul(add_trans, np.matmul(transf_holo2pv, transf_kinect2holo))
        if save_file["gender"][i] == 'm':
            smplx_model = smplx_male_model
        else:
            smplx_model = smplx_female_model
        body_params_dict_new = update_globalRT_for_smpl(body_param_dict=body_param_dict, smpl_model=smplx_model,
                                                        trans_to_target_origin=trans_to_target_origin, device=device)
        save_file['global_orient_pv'][i] = body_params_dict_new['global_orient'][0]
        save_file['transl_pv'][i] = body_params_dict_new['transl'][0]
        # import pdb; pdb.set_trace()
        #### get smpl 3d joints in pv coord
        for key in body_params_dict_new.keys():
            body_params_dict_new[key] = torch.FloatTensor(body_params_dict_new[key]).to(device)
        smpl_output_pv_coord = smplx_model(**body_params_dict_new)
        cur_joints_3d_pv = smpl_output_pv_coord.joints[:, 0:127].detach().cpu().numpy().squeeze()        
        save_file['3d_joints_pv'][i] = cur_joints_3d_pv         

    np.savez(os.path.join(save_path, save_name % split), **save_file)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='ProHMR training code')
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        help='split to process')
    args = parser.parse_args()

    for i in ["val", "train",  "test"]:
        preprocess(i, egobody_release_path)