import smplx
import torch
import pickle
import trimesh
import tqdm
import pyrender
import numpy as np
import subprocess
import cv2
import pdb
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from scipy.spatial.transform import Rotation
import pyquaternion as pyquat
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
jet = plt.get_cmap('twilight')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}

def rollout_primitives(motion_primitives):
    smplx_param_list = []
    gender = motion_primitives[0]['gender']

    if gender == 'male':
        body_model = bm_male20
    elif gender == 'female':
        body_model = bm_female20
    else:
        body_model = None
        pdb.set_trace()

    for idx, motion_primitive in enumerate(motion_primitives):
        pelvis_original = body_model(betas=torch.cuda.FloatTensor(motion_primitive['betas']).repeat(20, 1)).joints[:, 0, :].detach().cpu().numpy()  # [20, 3]
        smplx_param = motion_primitive['smplx_params'][0]  #[10, 96]

        rotation = motion_primitive['transf_rotmat'].reshape((3, 3)) # [3, 3]
        transl = motion_primitive['transf_transl'].reshape((1, 3)) # [1, 3]
        smplx_param[:, :3] = np.matmul((smplx_param[:, :3] + pelvis_original), rotation.T) - pelvis_original + transl
        r_ori = Rotation.from_rotvec(smplx_param[:, 3:6])
        r_new = Rotation.from_matrix(np.tile(motion_primitive['transf_rotmat'], [20, 1, 1])) * r_ori
        smplx_param[:, 3:6] = r_new.as_rotvec()

        if idx == 0:
            start_frame = 0
        elif motion_primitive['mp_type'] == '1-frame':
            start_frame = 1
        elif motion_primitive['mp_type'] == '2-frame':
            start_frame = 2
        else:
            # crowd-env use 1 frame model at the moment
            start_frame = 1
        smplx_param = smplx_param[start_frame:, :]
        smplx_param_list.append(smplx_param)

    return  np.concatenate(smplx_param_list, axis=0)  # [t, 96]


def gen_data_egobody(vis_marker=False, vis_pelvis=True, vis_object=False,
                vis_navmesh=True, start_frame=0,
                slow_rate=1, save_path=None, add_floor=True, scene_mesh=None):
    scene = pyrender.Scene()
    motions_list = []

    m = pyrender.Mesh.from_trimesh(scene_mesh)
    object_node = pyrender.Node(mesh=m, name='scene')
    scene.add_node(object_node)

    # eval model
    while True:
        # ret = subprocess.call(['python', 'crowd_ppo/main_egobody_eval.py', '--resume-path=/mnt/vlg-nfs/genli/log_pretrain_dep13_seedori/log_2f_ego_gru_rpene1_rlook0.3-finetune-newrpene0.1/collision-avoidance/ppo/0/231017-222547/policy.pth', '--watch', '--scene-name=%s' % scene_name])
        ret = subprocess.call(['python', 'crowd_ppo/main_egobody_eval.py', '--resume-path=data/checkpoint_best.pth', '--watch', '--scene-name=%s' % scene_name])
        if ret == 0:
            break
    result_paths = ['egobody_tmp_res/motion_0.pkl', 'egobody_tmp_res/motion_1.pkl']
    for input_path in result_paths:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            motions = data['motion']
            motions_list.append(motions)

    rollout_frames_list = [rollout_primitives(motions) for motions in motions_list]
    print(np.array([len(frames) for frames in rollout_frames_list]))
    max_frame = np.array([len(frames) for frames in rollout_frames_list]).max()

    rollout_frames_pad_list = []  # [T_max, 93], pad shorter sequences with last frame
    for idx in range(len(rollout_frames_list)):
        frames = rollout_frames_list[idx]
        rollout_frames_pad_list.append(np.concatenate([frames, np.tile(frames[-1:, :], (max_frame + 1 - frames.shape[0], 1))], axis=0))
    smplx_params = np.stack(rollout_frames_pad_list, axis=0)  # [S, T_max, 93]
    betas = [motions[0]['betas'] for motions in motions_list]
    betas = np.stack(betas, axis=0)  # [S, 10]
    genders = [motions[0]['gender'] for motions in motions_list]
    genders = np.stack(genders, axis=0)
    if genders[0] != genders[1]:
        pdb.set_trace()

    if genders[0] == 'male':
        body_model = bm_male2 
    elif genders[0] == 'female':
        body_model = bm_female2
    else:
        body_model = None
        pdb.set_trace()

    body_node = None
    camera_node = None

    renderer = pyrender.OffscreenRenderer(viewport_width=320, viewport_height=288)
    
    # Init camera here
    camera_pose = np.eye(4)
    camera = pyrender.camera.IntrinsicsCamera(fx=200, fy=200, cx=160, cy=144)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.)
    light_node = pyrender.Node(light=light)
    scene.add_node(light_node)

    for frame_idx in tqdm.tqdm(range(start_frame, max_frame)):

        # keep frames when social distance between 1m-3.5m
        flag_dist = False
        smplx_transl = smplx_params[:, frame_idx, :3]
        smplx_transl_dist = np.linalg.norm(smplx_transl[0] - smplx_transl[1])
        if smplx_transl_dist >= 1 and smplx_transl_dist <= 3.5:
            flag_dist = True
        else:
            continue

        smplx_dict = {
            'betas': betas,
            'transl': smplx_params[:, frame_idx, :3],
            'global_orient': smplx_params[:, frame_idx, 3:6],
            'body_pose': smplx_params[:, frame_idx, 6:69],
        }
        smplx_dict = params2torch(smplx_dict)

        output = body_model(**smplx_dict)
        vertices = output.vertices.detach().cpu().numpy()
        joints = output.joints.detach().cpu().numpy()

        body_meshes = []
        for seq_idx in range(vertices.shape[0]):
            m = trimesh.Trimesh(vertices=vertices[seq_idx], faces=body_model.faces, process=False)
            body_meshes.append(m)
        body_mesh = pyrender.Mesh.from_trimesh(body_meshes, smooth=False)
        # viewer.render_lock.acquire()
        if body_node is not None:
            scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=body_mesh, name='body')
        scene.add_node(body_node)
        # viewer.render_lock.release()

        for seq_idx in range(joints.shape[0]):
            flag_2d_joint = False
            flag_visible = False

            # only for one AR
            joint = joints[seq_idx]
            # 57: leye 56: reye
            # look_front. approx. may not be vertical to look_right
            look_at = joint[57] - joint[23] + joint[56] - joint[24]
            look_at = look_at.astype(np.float64)
            look_at = look_at / np.linalg.norm(look_at)
            # look_right
            leye_reye_dir = joint[23] - joint[24] 
            leye_reye_dir = leye_reye_dir.astype(np.float64)
            leye_reye_dir = leye_reye_dir / np.linalg.norm(leye_reye_dir)
            # look_up
            look_up_dir = np.cross(leye_reye_dir, look_at) 
            look_up_dir = look_up_dir.astype(np.float64)
            look_up_dir /= np.linalg.norm(look_up_dir)
            # only keep vertical componenet of look_at 
            look_at = np.cross(look_up_dir, leye_reye_dir)
            look_at = look_at.astype(np.float64)
            look_at /= np.linalg.norm(look_at)

            cam_pos = (joint[23] + joint[24]) / 2.
            # viewer.render_lock.acquire()
            if camera_node is not None:
                scene.remove_node(camera_node)
            up = np.array([0,1,0])
            front = np.array([0,0,-1])
            right = np.cross(up, front)
            look_at_up = np.cross(look_at, leye_reye_dir)
            look_at_up = look_at_up.astype(np.float64)
            look_at_up /= np.linalg.norm(look_at_up)
            r1 = np.stack([leye_reye_dir, look_at_up, look_at])
            r2 = np.stack([right, up, front])
            quat = pyquat.Quaternion(matrix=(r1.T @ r2))
            quat_pyrender = [quat[1], quat[2], quat[3], quat[0]]
            camera_node = pyrender.Node(camera=camera, name='camera', rotation=quat_pyrender, translation=cam_pos)
            scene.add_node(camera_node)
            # viewer.render_lock.release()

            # interactee joints projected to 2d camera plane, keep frames with >=6 joints visible
            interactee_joints_3d = joints[1-seq_idx][0:22]  # [22, 3] select 22 main body joints
            # project 3d joints to depth camera 2d plane
            cam_intrinsics = np.array([[200., 0., 160.],
                                     [0., 200., 144.],
                                     [0., 0., 1.]])
            camera_pose[:3, :3] = r1.T @ r2
            camera_pose[:3, 3] = cam_pos
            Rt = np.linalg.inv(camera_pose)
            interactee_joints_2d = cv2.projectPoints(interactee_joints_3d,
                                                     cv2.Rodrigues(Rt[:3, :3])[0], Rt[:3, 3],
                                                     cam_intrinsics,
                                                     np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))[0].squeeze()  # [22, 2]
            valid_x = np.logical_and(interactee_joints_2d[:, 1] >= (144-112), interactee_joints_2d[:, 1] <= (144+112))
            valid_y = np.logical_and(interactee_joints_2d[:, 0] >= (160-112), interactee_joints_2d[:, 0] <= (160+112))
            valid_joint_num = np.sum(valid_x * valid_y)
            if valid_joint_num >= 6:
                flag_2d_joint = True

            # discard all back-to-back frames
            look_at_2d = look_at[:2].astype(np.float64)
            look_at_2d /= np.linalg.norm(look_at_2d)
            dir_to_interactee = interactee_joints_3d[0][:2] - cam_pos[:2]
            dir_to_interactee = dir_to_interactee.astype(np.float64)
            dir_to_interactee /= np.linalg.norm(dir_to_interactee)
            if np.arccos(np.clip(np.dot(look_at_2d, dir_to_interactee), -1.0, 1.0)) < np.pi / 2:
                flag_visible = True

            if flag_dist and flag_2d_joint and flag_visible:
                rgb, depth = renderer.render(scene)
                global valid_num
                valid_num += 1
                # TODO: 1-seq_idx only correct for 2 human scenario
                base_path = 'tmp/egobody_depth'
                # if not os.path.exists(os.path.join(base_path, scene_name, 'rgb')):
                #     os.makedirs(os.path.join(base_path, scene_name, 'rgb'))
                if not os.path.exists(os.path.join(base_path, scene_name, 'depth_clean')):
                    os.makedirs(os.path.join(base_path, scene_name, 'depth_clean'))
                if not os.path.exists(os.path.join(base_path, scene_name, 'smplx_params')):
                    os.makedirs(os.path.join(base_path, scene_name, 'smplx_params'))
                # np.save(os.path.join(base_path, scene_name, 'rgb', '%d.npy' % valid_num), color)
                np.save(os.path.join(base_path, scene_name, 'depth_clean', '%d.npy' % valid_num), depth)

                # You may want to checkout what does it look like:
                # cv2.imwrite(os.path.join(base_path, scene_name, 'depth_clean', '%d.png' % valid_num), rgb)
                
                custom_smplx_params = np.zeros(96)
                custom_smplx_params[:69] = smplx_params[1 - seq_idx, frame_idx, :69]
                custom_smplx_params[69:85] = Rt.reshape(-1)
                custom_smplx_params[85:95] = betas[1 - seq_idx]
                custom_smplx_params[95] = 0 if genders[1 - seq_idx] == 'male' else 1
                np.save(os.path.join(base_path, scene_name, 'smplx_params', '%d.npy' % valid_num), custom_smplx_params)


# models folder contains: smplx/SMPLX_FEMALE.npz etc.
model_path = "data/smplx/models"
device = torch.device('cuda')
valid_num = 0

bm_male20 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="male",
                          use_pca=False,
                          batch_size=20,
                          ).to(device='cuda')
bm_female20 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="female",
                          use_pca=False,
                          batch_size=20,
                          ).to(device='cuda')

bm_male2 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="male",
                          use_pca=False,
                          batch_size=2,
                          ).to(device)
bm_female2 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="female",
                          use_pca=False,
                          batch_size=2,
                          ).to(device)

if __name__ == '__main__':
    MAX_NUM = 7000

    """
    with open('/mnt/vlg-nfs/kaizhao/datasets/scene_mesh_4render/list.txt', 'r') as f:
        scene_names = f.readlines()
    # only able to handle convex navmesh
    for scene_name in scene_names:
        scene_name = scene_name.strip()
        valid_num = 0
        scene_mesh = trimesh.load(os.path.join('/mnt/vlg-nfs/kaizhao/datasets/scene_mesh_4render', scene_name, 'mesh_floor_zup.ply'))
        while True:
            gen_data_egobody(
                scene_mesh = scene_mesh,
                vis_navmesh=False,
                vis_marker=False, vis_pelvis=False, vis_object=True, add_floor=False,
                )
            if valid_num >= MAX_NUM:
                break
    """

    scene_name = "seminar_d78"
    valid_num = 0
    scene_mesh = trimesh.load(os.path.join('exp_data', scene_name, 'mesh_floor_zup.ply'))
    while True:
        gen_data_egobody(
            scene_mesh = scene_mesh,
            vis_navmesh=False,
            vis_marker=False, vis_pelvis=False, vis_object=True, add_floor=False,
            )
        if valid_num >= MAX_NUM:
            break

