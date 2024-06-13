"""
Parts of the code are taken or adapted from
https://github.com/mkocabas/EpipolarPose/blob/master/lib/utils/img_utils.py
"""
import torch
import numpy as np
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode
from ..utils.camera import create_camera
import os
import PIL.Image as pil_img
from PIL import ImageDraw
import albumentations as A


def do_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis.
    """
    # np.random.seed(1)
    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.ROT_FACTOR if random.random() <= aug_config.ROT_AUG_RATE else 0
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    c_up = 1.0 + aug_config.COLOR_SCALE
    c_low = 1.0 - aug_config.COLOR_SCALE
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty

def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d: np.array, trans: np.array):
    """
    Transform a 2D point using translation matrix trans.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        trans (np.array): Transformation matrix.
    Returns:
        np.array: Transformed 2D point.
    """
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def generate_image_patch(img: np.array, c_x: float, c_y: float,
                         bb_width: float, bb_height: float,
                         patch_width: float, patch_height: float,
                         do_flip: bool, scale: float, rot: float) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1


    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img

def fliplr_params(smpl_params: Dict, has_smpl_params: Dict) -> Tuple[Dict, Dict]:
    """
    Flip SMPL parameters when flipping the image.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
    Returns:
        Dict, Dict: Flipped SMPL parameters and valid flags.
    """
    global_orient = smpl_params['global_orient'].copy()
    transl = smpl_params['transl'].copy()
    body_pose = smpl_params['body_pose'].copy()
    betas = smpl_params['betas'].copy()
    # has_global_orient = has_smpl_params['global_orient'].copy()
    # has_body_pose = has_smpl_params['body_pose'].copy()
    # has_betas = has_smpl_params['betas'].copy()
    has_global_orient = has_smpl_params['global_orient']
    has_transl = has_smpl_params['transl']
    has_body_pose = has_smpl_params['body_pose']
    has_betas = has_smpl_params['betas']

    body_pose_permutation = [6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                             14 ,18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                             34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                             45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                             56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
    body_pose_permutation = body_pose_permutation[:len(body_pose)]
    body_pose_permutation = [i-3 for i in body_pose_permutation]

    body_pose = body_pose[body_pose_permutation]

    global_orient[1::3] *= -1
    global_orient[2::3] *= -1
    body_pose[1::3] *= -1
    body_pose[2::3] *= -1

    smpl_params = {'global_orient': global_orient.astype(np.float32),
                   'body_pose': body_pose.astype(np.float32),
                   'betas': betas.astype(np.float32),
                   'transl': transl.astype(np.float32),
                  }

    has_smpl_params = {'global_orient': has_global_orient,
                       'body_pose': has_body_pose,
                       'betas': has_betas,
                       'transl': has_transl
                      }

    return smpl_params, has_smpl_params


def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]

    return joints

def keypoint_3d_processing(keypoints_3d: np.array, flip_permutation: List[int], rot: float, do_flip: float) -> np.array:
    """
    Process 3D keypoints (rotation/flipping).
    Args:
        keypoints_3d (np.array): Input array of shape (N, 4) containing the 3D keypoints and confidence.
        flip_permutation (List): Permutation to apply after flipping.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        np.array: Transformed 3D keypoints with shape (N, 4).
    """
    if do_flip:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    # in-plane rotation
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = -rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
    # keypoints_3d[:, :-1] = np.einsum('ij,kj->ki', rot_mat, keypoints_3d[:, :-1])
    keypoints_3d = np.einsum('ij,kj->ki', rot_mat, keypoints_3d)  # keypoints_3d: [n_joints, 3] rotate around z axis
    # flip the x coordinates
    keypoints_3d = keypoints_3d.astype('float32')
    return keypoints_3d



def rot_aa(aa: np.array, rot: float) -> np.array:
    """
    Rotate axis angle parameters.
    Args:
        aa (np.array): Axis-angle vector of shape (3,).
        rot (np.array): Rotation angle in degrees.
    Returns:
        np.array: Rotated axis-angle vector.
    """
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)

def smpl_param_processing(smpl_params: Dict, has_smpl_params: Dict, rot: float, do_flip: bool) -> Tuple[Dict, Dict]:
    """
    Apply random augmentations to the SMPL parameters.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        Dict, Dict: Transformed SMPL parameters and valid flags.
    """
    if do_flip:
        smpl_params, has_smpl_params = fliplr_params(smpl_params, has_smpl_params)
    smpl_params['global_orient'] = rot_aa(smpl_params['global_orient'], rot)
    return smpl_params, has_smpl_params

def rgb_processing(rgb_img_full):

    aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                A.Blur(blur_limit=(3, 10), p=0.1),
                A.RandomSnow(brightness_coeff=1.5,
                snow_point_lower=0.2, snow_point_upper=0.4)]
    aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(multiplier=[0.5, 1.5],
                elementwise=True, per_channel=True, p=0.2),
                A.HueSaturationValue(hue_shift_limit=20,
                sat_shift_limit=30, val_shift_limit=20,
                always_apply=False, p=0.2),
                A.Posterize(p=0.1),
                A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                A.Equalize(mode='cv', p=0.1)]
    albumentation_aug = A.Compose([A.OneOf(aug_comp), A.OneOf(aug_mod)])          
    rgb_img_full = albumentation_aug(image=rgb_img_full)['image']

    return rgb_img_full


def get_example(img_path: str, center_x: float, center_y: float,
                width: float, height: float,
                keypoints_2d: np.array, keypoints_3d: np.array,
                smpl_params: Dict, has_smpl_params: Dict,
                flip_2d_keypoint_permutation: List[int],
                flip_3d_keypoint_permutation: List[int],
                patch_width: int, patch_height: int,
                mean: np.array, std: np.array,
                do_augment: bool, augm_config: CfgNode, fx, cam_cx, cam_cy, is_train_synthetic = False) -> Tuple:
    """
    Get an example from the dataset and (possibly) apply random augmentations.
    Args:
        img_path (str): Image filename
        center_x (float): Bounding box center x coordinate in the original image.
        center_y (float): Bounding box center y coordinate in the original image.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array with shape (N,3) containing the 2D keypoints in the original image coordinates.
        keypoints_3d (np.array): Array with shape (N,4) containing the 3D keypoints.
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
        flip_kp_permutation (List): Permutation to apply to the keypoints after flipping.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        mean (np.array): Array of shape (3,) containing the mean for normalizing the input image.
        std (np.array): Array of shape (3,) containing the std for normalizing the input image.
        do_augment (bool): Whether to apply data augmentation or not.
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size
        img_patch (np.array): Cropped image patch of shape (3, patch_height, patch_height)
        keypoints_2d (np.array): Array with shape (N,3) containing the transformed 2D keypoints.
        keypoints_3d (np.array): Array with shape (N,4) containing the transformed 3D keypoints.
        smpl_params (Dict): Transformed SMPL parameters.
        has_smpl_params (Dict): Valid flag for transformed SMPL parameters.
        img_size (np.array): Image size of the original image.
        """
    #to determin if this image exist
    if not os.path.exists(img_path):
        #remove the first two folders in the img_path
        img_path = os.path.join(*img_path.split('/')[3:])
        img_path = '/' + img_path

    # 1. load image
    cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)
    img_height, img_width, img_channels = cvimg.shape

    img_size = np.array([img_height, img_width])

    ########## get augmentation params
    if do_augment:
        auge_scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        auge_scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = 1.0, 0, False, False, [1.0, 1.0, 1.0], 0., 0.

    # auge_scale = 1.2  # ok
    # rot=30  # ok
    # do_flip=True  # ok
    # tx, ty = 0.02, -0.02  # ok
    # # # tx, ty=0, 0

    center_x_orig, center_y_orgin, width_orgin, height_orgin = center_x.copy(), center_y.copy(), width.copy(), height.copy()
    if do_extreme_crop:
        center_x, center_y, width, height = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
    center_x += width * tx
    center_y += height * ty

    ######### Process 3D keypoints, only with rot, flip
    keypoints_3d_auge = keypoint_3d_processing(keypoints_3d, flip_3d_keypoint_permutation, rot, do_flip)

    ######### get aug cropped img patch
    img_patch_cv, trans_crop = generate_image_patch(cvimg,
                                               center_x, center_y,
                                               width, height,
                                               patch_width, patch_height,
                                               do_flip, auge_scale, rot)
    image = img_patch_cv.copy()
    image = image[:, :, ::-1]  # [224, 224, 3] BGR-->
    
    if is_train_synthetic:
        image = rgb_processing(image)

    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)

    # apply RGB normalization
    for n_c in range(img_channels):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]


    ######### Process smpl params, only with rot, flip
    smpl_params, has_smpl_params = smpl_param_processing(smpl_params, has_smpl_params, rot, do_flip)

    ######### get 2d keypoints for aug cropped img
    keypoints_2d_crop = keypoints_2d.copy()
    if do_flip:
        keypoints_2d_crop = fliplr_keypoints(keypoints_2d_crop, img_width, flip_2d_keypoint_permutation)
    keypoints_2d_crop_vis_mask = (keypoints_2d_crop[:, -1]>0).copy()  # [25]
    for n_jt in range(len(keypoints_2d)):
        keypoints_2d_crop[n_jt, 0:2] = trans_point2d(keypoints_2d_crop[n_jt, 0:2], trans_crop)
    # cv2.imshow('xxx', img_patch_cv)
    # cv2.waitKey(-1)
    temp = (keypoints_2d_crop[:, 0] >= 0) * (keypoints_2d_crop[:, 0] <= patch_width) * \
           (keypoints_2d_crop[:, 1] >= 0) * (keypoints_2d_crop[:, 1] <= patch_height)
    keypoints_2d_crop_vis_mask = keypoints_2d_crop_vis_mask * temp
    keypoints_2d_crop[:, :-1] = keypoints_2d_crop[:, :-1] / patch_width - 0.5  # normalize 2d joints to [-0.5~0.5]

    ######## the same augmentation result, but different orders:
    # flip orig img
    if do_flip:
        cvimg = cvimg[:, ::-1, :]
        center_x_auge = img_width - center_x - 1
    else:
        center_x_auge = center_x.copy()
    # rotate orig img around augmented (center_x_auge, center_y)
    M = cv2.getRotationMatrix2D((center_x_auge, center_y), rot, 1.0)

    # # # uncomment if need aug orig img in data loader
    # rotated_cvimg = cv2.warpAffine(cvimg, M, (1920, 1080))

    ############################### get augemented 3d in: auge crop cam and auge full cam
    # keypoints_3d_crop_auge = keypoints_3d.copy()
    # keypoints_3d_full_auge = keypoints_3d.copy()
    # cam_cx_auge = cam_cx.copy()
    keypoints_3d_full_noauge = keypoints_3d.copy()  # [24, 3]
    cam_t_full = [smpl_params['transl']]  # [1, 3]
    cam_t_full = np.asarray(cam_t_full)
    s = 2 * fx / (auge_scale * width) / cam_t_full[0, -1]
    delta_x = 2 * (center_x - cam_cx) / (auge_scale * width * s)
    delta_y = 2 * (center_y - cam_cy) / (auge_scale * width * s)
    cam_t_crop = [[cam_t_full[0, 0] - delta_x, cam_t_full[0, 1] - delta_y, auge_scale * width / 224 * cam_t_full[0, -1]]]
    # cam_t_crop = [[cam_t_full[0, 0] - delta_x, cam_t_full[0, 1] - delta_y, auge_scale * width / 224 * cam_t_full[0, -1] * 5000 / fx]]
    cam_t_crop = np.asarray(cam_t_crop)  # [1, 3]
    keypoints_3d_crop_noauge = keypoints_3d_full_noauge - cam_t_full + cam_t_crop  # 3d joints in cropped cam coord (before auge)

    keypoints_3d_crop_auge = keypoint_3d_processing(keypoints_3d_crop_noauge, flip_3d_keypoint_permutation, rot, do_flip=do_flip)  # 3d joints in cropped cam cord (after auge: rotate around z axis, flip)
    # keypoints_3d_full_auge = keypoints_3d_crop_auge - cam_t_crop + cam_t_full  # 3d joints in auged full cam coord (after auge), no flip
    if do_flip:
        cam_t_full[0, 0] = -cam_t_full[0, 0]
        cam_t_crop[0, 0] = -cam_t_crop[0, 0]
    keypoints_3d_full_auge = keypoints_3d_crop_auge - cam_t_crop + cam_t_full

    if do_flip:
        cam_cx_auge = 1920 - cam_cx
    else:
        cam_cx_auge = cam_cx.copy()



    rotated_img = 0
    rotated_cvimg = 0


    ######## get 2d keypoints for aug orig img
    keypoints_2d_aug_orig = keypoints_2d.copy()
    if do_flip:
        keypoints_2d_aug_orig = fliplr_keypoints(keypoints_2d_aug_orig, img_width, flip_2d_keypoint_permutation)
    for n_jt in range(len(keypoints_2d)):
        keypoints_2d_aug_orig[n_jt, 0:2] = trans_point2d(keypoints_2d_aug_orig[n_jt, 0:2], M)


    # normalize 2d joints to [-0.5~0.5]
    keypoints_2d_aug_orig[:, 0] = keypoints_2d_aug_orig[:, 0] / img_width - 0.5
    keypoints_2d_aug_orig[:, 1] = keypoints_2d_aug_orig[:, 1] / img_height - 0.5


    return img_patch, keypoints_2d_crop, keypoints_2d_crop_vis_mask, \
           keypoints_3d_crop_auge, keypoints_3d_full_auge, smpl_params, has_smpl_params, img_size, img_patch_cv, \
           center_x_auge, center_y, cam_cx_auge, auge_scale, keypoints_2d_aug_orig, rotated_img, rotated_cvimg



def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25+0, 25+1, 25+4, 25+5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height

def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height

def full_body(keypoints_2d: np.array) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0).sum() == len(body_keypoints)

def upper_body(keypoints_2d: np.array):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25+8, 25+9, 25+12, 25+13, 25+17, 25+18]
    return ((keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0)\
       and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)

def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.array): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale

def extreme_cropping(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)

    return center_x, center_y, max(width, height), max(width, height)
