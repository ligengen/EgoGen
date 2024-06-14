"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
import copy

# def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
#     """
#     Computes a similarity transform (sR, t) in a batched way that takes
#     a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
#     where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
#     i.e. solves the orthogonal Procrutes problem.
#     Args:
#         S1 (torch.Tensor): First set of points of shape (B, N, 3).
#         S2 (torch.Tensor): Second set of points of shape (B, N, 3).
#     Returns:
#         (torch.Tensor): The first set of points after applying the similarity transformation.
#     """
#
#     batch_size = S1.shape[0]
#     S1 = S1.permute(0, 2, 1)
#     S2 = S2.permute(0, 2, 1)
#     # 1. Remove mean.
#     mu1 = S1.mean(dim=2, keepdim=True)
#     mu2 = S2.mean(dim=2, keepdim=True)
#     X1 = S1 - mu1
#     X2 = S2 - mu2
#
#     # 2. Compute variance of X1 used for scale.
#     var1 = (X1**2).sum(dim=(1,2))
#
#     # 3. The outer product of X1 and X2.
#     K = torch.matmul(X1, X2.permute(0, 2, 1))
#
#     # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
#     U, s, V = torch.svd(K)
#     Vh = V.permute(0, 2, 1)
#
#     # Construct Z that fixes the orientation of R to get det(R)=1.
#     Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
#     Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))
#
#     # Construct R.
#     R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))
#
#     # 5. Recover scale.
#     trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
#     scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)
#
#     # 6. Recover translation.
#     t = mu2 - scale*torch.matmul(R, mu1)
#
#     # 7. Error:
#     S1_hat = scale*torch.matmul(R, S1) + t
#
#     return S1_hat.permute(0, 2, 1)


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def compute_similarity_transform_batch_with_vis_mask(vis_mask, S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform_with_vis_mask(vis_mask[i], S1[i], S2[i])
    return S1_hat

def compute_similarity_transform_with_vis_mask(vis_mask, S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1_copy = S1*vis_mask
        S2_copy = S2*vis_mask
        S1_copy = S1_copy.T
        S2_copy = S2_copy.T
        transposed = True
    assert(S2_copy.shape[1] == S1_copy.shape[1])

    mu1 = S1_copy.mean(axis=1, keepdims=True)
    mu2 = S2_copy.mean(axis=1, keepdims=True)
    X1 = S1_copy - mu1
    X2 = S2_copy - mu2
    var1 = np.sum(X1**2)
    K = X1.dot(X2.T)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1.T) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def reconstruction_error(S1, S2, avg_joint=True, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))

    if avg_joint:
        re = re.mean(axis=-1)
    # if reduction == 'mean':
    #     re = re.mean()
    # elif reduction == 'sum':
    #     re = re.sum()

    return re

def reconstruction_error_with_vis_mask(vis_mask, S1, S2, avg_joint=True, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    # vis_mask: [bs*n_samples, n_joints]
    # S1/S2: [bs*n_samples, n_joints, 3]
    S1_hat = compute_similarity_transform_batch_with_vis_mask(vis_mask, S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))

    if avg_joint:
        re = re.mean(axis=-1)
    # if reduction == 'mean':
    #     re = re.mean()
    # elif reduction == 'sum':
    #     re = re.sum()

    return re

# def reconstruction_error(S1, S2) -> np.array:
#     """
#     Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
#     Args:
#         S1 (torch.Tensor): First set of points of shape (B, N, 3).
#         S2 (torch.Tensor): Second set of points of shape (B, N, 3).
#     Returns:
#         (np.array): Reconstruction error.
#     """
#     S1_hat = compute_similarity_transform(S1, S2)
#     re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
#     return re.cpu().numpy()

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstuction_error
    r_error = reconstruction_error(pred_joints.cpu(), gt_joints.cpu())
    return 1000 * mpjpe, 1000 * r_error



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

def row(A):
    return A.reshape((1, -1))

def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord

class Evaluator:

    def __init__(self,
                 dataset_length: int,
                 keypoint_list: List,
                 pelvis_ind: int, 
                 metrics: List = ['mode_mpjpe', 'mode_re', 'min_mpjpe', 'min_re']):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.keypoint_list = keypoint_list
        self.pelvis_ind = pelvis_ind
        self.metrics = metrics
        for metric in self.metrics:
            setattr(self, metric, np.zeros((dataset_length,)))
        self.counter = 0

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} / {self.dataset_length} samples')
        for metric in self.metrics:
            print(f'{metric}: {getattr(self, metric)[:self.counter].mean()} mm')
        print('***')
    
    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
            opt_output (Dict): Optimization output.
        """
        pred_keypoints_3d = output['pred_keypoints_3d'].detach()
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)

        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]

        # Compute joint errors
        mpjpe, re = eval_pose(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[:, self.keypoint_list], gt_keypoints_3d.reshape(batch_size * num_samples, -1 ,3)[:, self.keypoint_list])
        mpjpe = mpjpe.reshape(batch_size, num_samples)
        re = re.reshape(batch_size, num_samples)

        # Compute joint errors after optimization, if available.
        if opt_output is not None:
            opt_keypoints_3d = opt_output['model_joints']
            opt_keypoints_3d -= opt_keypoints_3d[:, [self.pelvis_ind]]
            opt_mpjpe, opt_re = eval_pose(opt_keypoints_3d[:, self.keypoint_list], gt_keypoints_3d[:, 0, self.keypoint_list])
        
        # The 0-th sample always corresponds to the mode
        if hasattr(self, 'mode_mpjpe'):
            mode_mpjpe = mpjpe[:, 0]
            self.mode_mpjpe[self.counter:self.counter+batch_size] = mode_mpjpe
        if hasattr(self, 'mode_re'):
            mode_re = re[:, 0]
            self.mode_re[self.counter:self.counter+batch_size] = mode_re
        if hasattr(self, 'min_mpjpe'):
            min_mpjpe = mpjpe.min(axis=-1)
            self.min_mpjpe[self.counter:self.counter+batch_size] = min_mpjpe
        if hasattr(self, 'min_re'):
            min_re = re.min(axis=-1)
            self.min_re[self.counter:self.counter+batch_size] = min_re
        if hasattr(self, 'opt_mpjpe'):
            self.opt_mpjpe[self.counter:self.counter+batch_size] = opt_mpjpe
        if hasattr(self, 'opt_re'):
            self.opt_re[self.counter:self.counter+batch_size] = opt_re

        self.counter += batch_size


