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

# replace the following path with the path to the egogen rgb folder
data_root = ""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
        joints_3d_kinect = smpl_out.joints[:, 0:24].detach().cpu().numpy()  # [1, 24, 3]

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

def get_center_scale(keypoints, confidence_thresh=0.2, rescale=1.2):
    valid = keypoints[:, 2] > confidence_thresh
    valid_keypoints = keypoints[valid][:, :-1]

    center = valid_keypoints.mean(axis=0)
    # center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2
    
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

if __name__ == "__main__":

    scene_name_list = ['cab_e', 'cab_g_benches', 'cab_h_tables', 'cnb_dlab_0215', 'cnb_dlab_0225', 'foodlab_0312',
                       'kitchen_gfloor', 'seminar_d78', 'seminar_d78_0318', 'seminar_g110', 'seminar_g110_0315',
                       'seminar_g110_0415', 'seminar_h52', 'seminar_h53_0218', 'seminar_j716']

    out_dict = {'imgname': [],
                'blurimgname': [],
                'valid_keypoints': [],
                'gender': [],
                '3d_joints_pv': [],
                'shape': [],
                'global_orient_pv': [],
                'transl_pv': [],
                'pose': [],
                'center': [],
                'scale': [],
                'cx': [],
                'cy': [],
                'fx': [],
                'fy': []
                }

    smplx_male = smplx.create('../data/smplx_model', model_type='smplx', gender='male', batch_size=1, ext='npz').to(device)
    smplx_female = smplx.create('../data/smplx_model', model_type='smplx', gender='female', batch_size=1, ext='npz').to(device)

    opengv_opencv_trans = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    for scene_name in scene_name_list:
        path = os.path.join(data_root, scene_name)
        num_frames = len(glob.glob1(path+"/rgb","*.jpg"))
        print("There are %d frames in scene %s"%(num_frames, scene_name))

        for frame_id in tqdm(range(1594, num_frames+1)):
        # for frame_id in tqdm(range(1594, 2000)):
            smplx_params = np.load(os.path.join(data_root, scene_name, 'smplx_params', '{}.npy'.format(frame_id)))  # [93]
            smplx_params_dict = {}
            smplx_params_dict['transl'] = smplx_params[0:3]
            smplx_params_dict['global_orient'] = smplx_params[3:6]
            smplx_params_dict['body_pose'] = smplx_params[6:69]
            extrinsic = smplx_params[69:85].reshape(4, 4)
            extrinsic = np.matmul(opengv_opencv_trans, extrinsic)
            smplx_params_dict['betas'] = smplx_params[85:95]
            gender = smplx_params[95]  # 0 male, 1 female
            cx = smplx_params[96]
            cy = smplx_params[97]
            fx = smplx_params[98]
            fy = fx
            smplx_model = smplx_male if gender == 0 else smplx_female

            for key in smplx_params_dict.keys():
                smplx_params_dict[key] = np.expand_dims(smplx_params_dict[key], axis=0)
            ####### transform smplx params to camera frame coordinate system
            smplx_params_dict = update_globalRT_for_smpl(body_param_dict=smplx_params_dict, smpl_model=smplx_model,
                                                         trans_to_target_origin=extrinsic, device=device)

            smplx_params_rgb_torch = {}
            for key in smplx_params_dict.keys():
                smplx_params_rgb_torch[key] = torch.from_numpy(smplx_params_dict[key]).float().to(device)
            smplx_body = smplx_model(**smplx_params_rgb_torch)
            vertices = smplx_body.vertices
            joints_rgb = smplx_body.joints[0].detach().cpu().numpy()

            out_dict['imgname'].append('/'.join([scene_name, 'rgb', '{}.jpg'.format(frame_id)]))
            out_dict['blurimgname'].append('/'.join([scene_name, 'rgb_blur', '{}.jpg'.format(frame_id)]))
            if gender == 0:
                gender = 'm'
            else:
                gender = 'f'
            out_dict['gender'].append(gender)
            out_dict['3d_joints_pv'].append(joints_rgb)
            out_dict['shape'].append(smplx_params_dict['betas'][0])
            out_dict['global_orient_pv'].append(smplx_params_dict['global_orient'][0])
            out_dict['transl_pv'].append(smplx_params_dict['transl'][0])
            out_dict['pose'].append(smplx_params_dict['body_pose'][0])
            out_dict['cx'].append(cx)
            out_dict['cy'].append(cy)
            out_dict['fx'].append(fx)
            out_dict['fy'].append(fy)

            cam_intrinsics = np.array([[fx, 0., cx],
                                     [0., fx, cy],
                                     [0., 0., 1.]])
            valid_keypoints = cv2.projectPoints(joints_rgb,
                                                 cv2.Rodrigues(np.eye(3))[0], np.array([0., 0., 0.]),
                                                 cam_intrinsics,
                                                 np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))[0].squeeze()  # [22, 2]
            
            # change the order of the joints and add confidence score
            smplx_to_openpose = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
            valid_keypoints = valid_keypoints[smplx_to_openpose]
            valid_keypoints = np.concatenate([valid_keypoints, np.ones([25, 1])], axis=1)
            valid_keypoints[valid_keypoints[:, 0] < 0, 2] = 0
            valid_keypoints[valid_keypoints[:, 0] > 1919, 2] = 0
            valid_keypoints[valid_keypoints[:, 1] < 0, 2] = 0
            valid_keypoints[valid_keypoints[:, 1] > 1079, 2] = 0


            out_dict['valid_keypoints'].append(valid_keypoints)
            
            center, scale = get_center_scale(valid_keypoints)
            out_dict['center'].append(center)
            out_dict['scale'].append(scale)
        


    for key in out_dict.keys():
        out_dict[key] = np.asarray(out_dict[key])

    if not os.path.exists(os.path.join(data_root, "smplx_spin_holo_rgb_npz")):
        os.makedirs(os.path.join(data_root, "smplx_spin_holo_rgb_npz"))

    np.savez(os.path.join(data_root, 'smplx_spin_holo_rgb_npz', 'egocapture_train_smplx.npz'), **out_dict)


    print('npz saved.')
    print('# of frames:', len(out_dict['imgname']))
    

