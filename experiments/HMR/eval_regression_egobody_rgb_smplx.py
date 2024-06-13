"""
Script used for evaluating the 3D pose errors of ProHMR (mode + minimum).

Example usage:
python eval_regression.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import torch
import argparse
from tqdm import tqdm
from prohmr.configs import get_config, prohmr_config, dataset_config
# from prohmr.models import ProHMR
from prohmr.utils import Evaluator, recursive_to
# from prohmr.datasets import create_dataset
import smplx
import numpy as np
import PIL.Image as pil_img
import open3d as o3d
import json
import copy
import pickle as pkl
import random

from prohmr.models import ProHMRRGBSmplx
from prohmr.utils.pose_utils import reconstruction_error
from prohmr.utils.renderer import *

from prohmr.datasets.image_dataset_rgb_egobody_smplx import ImageDatasetEgoBodyRgbSmplx
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis

# python eval_regression_egobody_rgb_smplx.py --with_focal_length True --with_bbox_info True --with_cam_center True --batch_size 1 --vis_freq 1
# python eval_regression_egobody_rgb_smplx.py --checkpoint ./data/checkpoint/rgb/best_model.pt --dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release/

parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='runs_egogen_rgb/32188/best_model.pt')  # runs_try/90505/best_model.pt data/checkpoint.pt
parser.add_argument('--model_cfg', type=str, default='prohmr/configs/prohmr.yaml', help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
# parser.add_argument('--dataset', type=str, default='ethfv_smpl', choices=['H36M-VAL-P2', '3DPW-TEST', 'MPI-INF-TEST'], help='Dataset to evaluate')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=100, help='How often to log results')
# parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
parser.add_argument('--eval_freq', type=int, default=1, help='calculate loss freq')  # todo: not setup correctly
parser.add_argument('--output_vis_folder', default='output_vis/output_32188_rotate_backup', help='output folder')  # output_49001
parser.add_argument('--vis', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--vis_multi_sample', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--vis_freq', type=int, default=32, help='only set to >1 when need visualize')

parser.add_argument('--add_bbox_scale', type=float, default=1.2, help='add scale to orig bbox')

parser.add_argument('--save_results', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--output_root', type=str, default='output_results_try')  # output_results_final
parser.add_argument("--seed", default=0, type=int)

parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'])

parser.add_argument('--with_vfov', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_joint_vis', default='False', type=lambda x: x.lower() in ['true', '1'])

parser.add_argument('--err_multi_mode', default='False', type=lambda x: x.lower() in ['true', '1'])

parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'])  # todo

parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')

parser.add_argument('--dataset_root', type=str, default="", help='path to the egobody folder')

args = parser.parse_args()

# Use the GPU if available

torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())

######## todo uncomment if want to render, not compatible with open3d visualization
import pyrender
H, W = 350, 388
camera_pose = np.eye(4)
camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
camera = pyrender.camera.IntrinsicsCamera(
    fx=200, fy=200,
    cx=int(W/2), cy=int(H/2))
light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
base_color = (1.0, 193/255, 193/255, 1.0)
material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=base_color
    )
r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
import matplotlib.pyplot as plt
cmap= plt.get_cmap('turbo')  # viridis
color_map = cmap.colors
color_map = np.asarray(color_map)


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
fixseed(args.seed)

# Load model config
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()


# Setup model
# model = ProHMRRGBSmplx(cfg=model_cfg, device=device, with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                    #   with_vfov=args.with_vfov, with_joint_vis=args.with_joint_vis)
model = ProHMRRGBSmplx(cfg=model_cfg, device=device, writer=None, logger=None,
                          with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                          with_vfov=args.with_vfov, with_joint_vis=args.with_joint_vis)
weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
# import pdb; pdb.set_trace()
model.load_state_dict(weights['state_dict'])
model.eval()
print(args.checkpoint)


# Create dataset and data loader
# dataset = create_dataset(model_cfg, dataset_cfg, train=False, add_scale=args.add_scale)
test_dataset = ImageDatasetEgoBodyRgbSmplx(cfg=model_cfg, train=False, device=device, img_dir=args.dataset_root,
                                      dataset_file=args.dataset_root.split('egobody_release')[0] +'smplx_spin_npz/egocapture_test_smplx.npz',
                                      spacing=1,    add_scale=args.add_bbox_scale)
dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

smpl_neutral = smplx.create('data/smplx_model', model_type='smplx', create_transl=False, batch_size=args.batch_size).to(device)
smpl_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', batch_size=args.batch_size).to(device)
smpl_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', batch_size=args.batch_size).to(device)

# eval_sample_n = len(dataset) // args.eval_freq
g_mpjpe = np.zeros(len(test_dataset))
mpjpe = np.zeros(len(test_dataset))
pa_mpjpe = np.zeros(len(test_dataset))
g_v2v = np.zeros(len(test_dataset))
v2v = np.zeros(len(test_dataset))  # translation/pelv-aligned
pa_v2v = np.zeros(len(test_dataset))  # procrustes aligned
img_name_list = []

if args.vis:
    os.mkdir(args.output_vis_folder) if not os.path.exists(args.output_vis_folder) else None
    renderer = Renderer(model_cfg, faces=smpl_neutral.faces)


pred_betas_list = []
pred_body_pose_list = []
pred_global_orient_list = []
pred_cam_list = []
pred_cam_full_list = []

v2v_list = {}

# Go over the images in the dataset.
for step, batch in enumerate(tqdm(dataloader)):
    if step % args.eval_freq == args.eval_freq - 1:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)


        gt_pose = {}
        gt_pose['global_orient'] = batch['smpl_params']['global_orient'].to(device)
        gt_pose['transl'] = batch['smpl_params']['transl'].to(device)
        gt_pose['body_pose'] = batch['smpl_params']['body_pose'].to(device)
        gt_pose['betas'] = batch['smpl_params']['betas'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = batch['img'].shape[0]
        img_names = batch['imgname']
        bbox_size = batch['box_size'].to(device)  # [bs]
        bbox_center = batch['box_center'].to(device)  # [bs, 2]
        focal_length = batch['fx'] * model_cfg.CAM.FX_NORM_COEFF  # [bs]
        cam_cx = batch['cam_cx']
        cam_cy = batch['cam_cy']

        pred_betas = out['pred_smpl_params']['betas']  #  [bs, num_sample, 10]
        pred_body_pose = out['pred_smpl_params']['body_pose']  # [bs, num_sample, 23, 3, 3]
        pred_global_orient = out['pred_smpl_params']['global_orient']  # [bs, num_sample, 1, 3, 3]
        pred_body_pose = rotation_matrix_to_angle_axis(pred_body_pose.reshape(-1, 3, 3)).reshape(curr_batch_size, args.num_samples, -1, 3)
        pred_global_orient = rotation_matrix_to_angle_axis(pred_global_orient.reshape(-1, 3, 3)).reshape(curr_batch_size, args.num_samples, -1, 3)
        pred_cam = pred_transl = out['pred_cam']  #  [bs, num_sample, 3]


        ##############
        if curr_batch_size != args.batch_size:
            smpl_neutral = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', batch_size=curr_batch_size * args.num_samples, ext='npz').to(device)
            smpl_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', batch_size=curr_batch_size * args.num_samples, ext='npz').to(device)
            smpl_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', batch_size=curr_batch_size * args.num_samples, ext='npz').to(device)
        pred_output = smpl_neutral(betas=pred_betas.reshape(-1, 10), body_pose=pred_body_pose.reshape(-1, 21, 3),
                                    global_orient=pred_global_orient.reshape(-1, 1, 3))
        pred_vertices = pred_output.vertices.reshape(curr_batch_size, -1, 10475, 3)
        pred_keypoints_3d = pred_output.joints.reshape(curr_batch_size, -1, 127, 3)[:, :, :22, :]  # [bs, n_sample, 22, 3]
        pred_vertices_mode = pred_vertices[:, 0]
        pred_keypoints_3d_mode = pred_keypoints_3d[:, 0]  # [bs, 22, 3]
        pred_keypoints_3d_global_mode = pred_keypoints_3d_mode + pred_transl[:, 0].unsqueeze(-2)
        pred_vertices_global_mode = pred_vertices_mode + pred_transl[:, 0].unsqueeze(-2)

        pred_betas_list.append(pred_betas)
        pred_body_pose_list.append(pred_body_pose)
        pred_global_orient_list.append(pred_global_orient)

        ###### single mode with z_0
        pred_pelvis_mode = pred_keypoints_3d_mode[:, [0], :].clone()
        pred_keypoints_3d_mode_align = pred_keypoints_3d_mode - pred_pelvis_mode
        pred_vertices_mode_align = pred_vertices_mode - pred_pelvis_mode
        pred_cam_mode = pred_cam[:, 0]

        ##### get pred cam in full img coord
        pred_cam_full = convert_pare_to_full_img_cam(pare_cam=pred_cam_mode, bbox_height=bbox_size,
                                                     bbox_center=bbox_center,
                                                     # img_w=1920, img_h=1080,
                                                     img_w=cam_cx * 2, img_h=cam_cy * 2,
                                                     focal_length=focal_length,
                                                     crop_res=model_cfg.MODEL.IMAGE_SIZE)  # [bs, 3]
        pred_vertices_full = pred_vertices_mode + pred_cam_full.unsqueeze(1)  # [bs, 6890, 3]
        pred_keypoints_3d_full = pred_keypoints_3d_mode + pred_cam_full.unsqueeze(1)  # [bs, 24, 3]

        pred_cam_full_list.append(pred_cam_full)

        ##### get gt body
        gt_body = smpl_male(**gt_pose)
        gt_joints = gt_body.joints
        gt_vertices = gt_body.vertices
        gt_body_female = smpl_female(**gt_pose)
        gt_joints_female = gt_body_female.joints
        gt_vertices_female = gt_body_female.vertices
        gt_joints[gender == 1, :, :] = gt_joints_female[gender == 1, :, :]
        gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]

        gt_keypoints_3d = gt_joints[:, :22, :]  # [bs, 24, 3]
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()  # [bs,1,3]
        gt_keypoints_3d_align = gt_keypoints_3d - gt_pelvis
        gt_vertices_align = gt_vertices - gt_pelvis

        # G-MPJPE
        error_per_joint = torch.sqrt(((pred_keypoints_3d_full - gt_keypoints_3d) ** 2).sum(dim=-1))
        error = error_per_joint.mean(dim=-1).cpu().detach().numpy()
        g_mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # MPJPE
        error_per_joint = torch.sqrt(((pred_keypoints_3d_mode_align - gt_keypoints_3d_align) ** 2).sum(dim=-1))
        error = error_per_joint.mean(dim=-1).cpu().detach().numpy()
        mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # PA-MPJPE
        # import pdb;  pdb.set_trace()
        pa_error_per_joint = reconstruction_error(pred_keypoints_3d_mode_align.cpu().detach().numpy(), gt_keypoints_3d_align.cpu().detach().numpy(), avg_joint=False)  # [bs, n_joints]
        pa_error = pa_error_per_joint.mean(axis=-1)  # [bs]
        pa_mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = pa_error

        # G-V2V
        error_per_verts = torch.sqrt(((pred_vertices_full - gt_vertices) ** 2).sum(dim=-1))
        error = error_per_verts.mean(dim=-1).cpu().detach().numpy()
        g_v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # V2V
        error_per_verts = torch.sqrt(((pred_vertices_mode_align - gt_vertices_align) ** 2).sum(dim=-1))
        error = error_per_verts.mean(dim=-1).cpu().detach().numpy()
        v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # PA-V2V
        pa_error = reconstruction_error(pred_vertices_mode_align.cpu().detach().numpy(), gt_vertices_align.cpu().detach().numpy(), avg_joint=True)
        pa_v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = pa_error

        # add img_name to list
        img_name_list.extend(img_names)

        if step % args.log_freq == 0 and step > 0:
            print('G-MPJPE: ' + str(1000 * g_mpjpe[:step * args.batch_size].mean()))
            print('MPJPE: ' + str(1000 * mpjpe[:step * args.batch_size].mean()))
            print('PA-MPJPE: ' + str(1000 * pa_mpjpe[:step * args.batch_size].mean()))
            print('G-V2V: ' + str(1000 * g_v2v[:step * args.batch_size].mean()))
            print('V2V: ' + str(1000 * v2v[:step * args.batch_size].mean()))
            print('PA-V2V: ' + str(1000 * pa_v2v[:step * args.batch_size].mean()))

        ######################## visualize 3d bodies and scene in the original physical camera
        # if args.vis and step % args.vis_freq == 0:

        #     save_root = args.output_vis_folder + '/rgb/'
        #     os.makedirs(save_root) if not os.path.exists(save_root) else None
        #     rgb_save_path = os.path.join(save_root, '{}_{}'.format(batch['imgname'][0].split('/')[-4], batch['imgname'][0].split('/')[-1].replace('jpg', 'png')))
        #     curr_image = batch['imgname'][0]
        #     curr_image = cv2.imread(curr_image)
        #     cv2.imwrite(rgb_save_path, curr_image)

        #     body = trimesh.Trimesh(gt_vertices[0].detach().cpu().numpy(), smpl_neutral.faces, process=False)
        #     gt_body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
        #     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        #     scene.add(camera, pose=camera_pose)
        #     scene.add(light, pose=camera_pose)
        #     scene.add(gt_body_mesh, 'mesh')
        #     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        #     color = color.astype(np.float32) / 255.0
        #     alpha = 1.0  # set transparency in [0.0, 1.0]
        #     color[:, :, -1] = color[:, :, -1] * alpha
        #     color = pil_img.fromarray((color * 255).astype(np.uint8))
        #     save_root = args.output_vis_folder + '/gt/'
        #     os.makedirs(save_root) if not os.path.exists(save_root) else None
        #     gt_save_path = os.path.join(save_root, '{}_{}'.format(batch['imgname'][0].split('/')[-4], batch['imgname'][0].split('/')[-1].replace('jpg', 'png')))
        #     color.save(gt_save_path)
        
        #     ####### render with error color map
        #     error = np.linalg.norm(np.abs(gt_vertices_align[0].detach().cpu().numpy() - pred_vertices_mode_align[0].detach().cpu().numpy()), axis=-1)  # [10475]
        #     error = error * 1000 / 120
        #     error = (error * 255).astype(int)
        #     error[error>255] = 255
        #     error = color_map[error]  # in [0, 1]
        #     body = trimesh.Trimesh(vertices=pred_vertices_full[0].detach().cpu().numpy(), vertex_colors=error, faces=smpl_neutral.faces, process=False)
        #     body_mesh = pyrender.Mesh.from_trimesh(body)
        #     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        #     scene.add(camera, pose=camera_pose)
        #     scene.add(light, pose=camera_pose)
        #     scene.add(body_mesh, 'mesh')
        #     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        #     color = color.astype(np.float32) / 255.0
        #     alpha = 1.0  # set transparency in [0.0, 1.0]
        #     color[:, :, -1] = color[:, :, -1] * alpha
        #     color = pil_img.fromarray((color * 255).astype(np.uint8))
        #     save_root = args.output_vis_folder + '/pred/'
        #     os.makedirs(save_root) if not os.path.exists(save_root) else None
        #     pred_save_path = os.path.join(save_root, '{}_{}'.format(batch['imgname'][0].split('/')[-4], batch['imgname'][0].split('/')[-1].replace('jpg', 'png')))
        #     color.save(pred_save_path)
            
        #     v2v_list[batch['imgname'][0].split('/')[-1]] = (v2v[step * args.batch_size], rgb_save_path, gt_save_path, pred_save_path)
        

        ####################### visualize 3d bodies with rotation
        # if args.vis and step % args.vis_freq == 0:

        #     save_root = args.output_vis_folder + '/rgb/'
        #     os.makedirs(save_root) if not os.path.exists(save_root) else None
        #     rgb_save_path = os.path.join(save_root, '{}_{}'.format(batch['imgname'][0].split('/')[-4], batch['imgname'][0].split('/')[-1].replace('jpg', 'png')))
        #     curr_image = batch['imgname'][0]
        #     curr_image = cv2.imread(curr_image)
        #     cv2.imwrite(rgb_save_path, curr_image)

        #     for rot_angle in range(0, 360, 1):
        #         gt_vertices_align_rot = gt_vertices_align[0].detach().cpu().numpy().copy()
        #         rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
        #         gt_vertices_align_rot = np.dot(gt_vertices_align_rot, rot[:3, :3].T)
        #         gt_vertices_rot = gt_vertices_align_rot + gt_pelvis[0].detach().cpu().numpy()
        #         body = trimesh.Trimesh(gt_vertices_rot, smpl_neutral.faces, process=False)


        #         gt_body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
        #         scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        #         scene.add(camera, pose=camera_pose)
        #         scene.add(light, pose=camera_pose)
        #         scene.add(gt_body_mesh, 'mesh')
        #         color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        #         color = color.astype(np.float32) / 255.0
        #         alpha = 1.0  # set transparency in [0.0, 1.0]
        #         color[:, :, -1] = color[:, :, -1] * alpha
        #         color = pil_img.fromarray((color * 255).astype(np.uint8))
        #         save_root = args.output_vis_folder + '/gt/' + '{}_{}'.format(batch['imgname'][0].split('/')[-4], batch['imgname'][0].split('/')[-1].replace('.jpg', ''))
        #         os.makedirs(save_root) if not os.path.exists(save_root) else None
        #         # output with name 3 digits
        #         out_angle = str(rot_angle).zfill(3)
        #         gt_save_path = os.path.join(save_root, 'image-{}.png'.format(rot_angle))
        #         color.save(gt_save_path)
                
        
        #     ####### render with error color map
        #     for rot_angle in range(0, 360, 1):
        #         error = np.linalg.norm(np.abs(gt_vertices_align[0].detach().cpu().numpy() - pred_vertices_mode_align[0].detach().cpu().numpy()), axis=-1)  # [10475]
        #         error = error * 1000 / 120
        #         error = (error * 255).astype(int)
        #         error[error>255] = 255
        #         error = color_map[error]  # in [0, 1]

        #         pred_vertices_align_rot = pred_vertices_mode_align[0].detach().cpu().numpy().copy()
        #         rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
        #         pred_vertices_align_rot = np.dot(pred_vertices_align_rot, rot[:3, :3].T)
        #         pred_vertices_rot = pred_vertices_align_rot + gt_pelvis[0].detach().cpu().numpy()
                
        #         body = trimesh.Trimesh(vertices=pred_vertices_rot, vertex_colors=error, faces=smpl_neutral.faces, process=False)
        #         body_mesh = pyrender.Mesh.from_trimesh(body)
        #         scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        #         scene.add(camera, pose=camera_pose)
        #         scene.add(light, pose=camera_pose)
        #         scene.add(body_mesh, 'mesh')
        #         color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        #         color = color.astype(np.float32) / 255.0
        #         alpha = 1.0  # set transparency in [0.0, 1.0]
        #         color[:, :, -1] = color[:, :, -1] * alpha
        #         color = pil_img.fromarray((color * 255).astype(np.uint8))
        #         save_root = args.output_vis_folder + '/pred/' + '{}_{}'.format(batch['imgname'][0].split('/')[-4], batch['imgname'][0].split('/')[-1].replace('.jpg', ''))
        #         os.makedirs(save_root) if not os.path.exists(save_root) else None
        #         out_angle = str(rot_angle).zfill(3)
        #         pred_save_path = os.path.join(save_root, 'image-{}.png'.format(rot_angle))
        #         color.save(pred_save_path)
            
            # break

print('*** Final Results ***')
print('[cfg] err_multi_mode: ', args.err_multi_mode)
print('G-MPJPE: ' + str(1000 * g_mpjpe.mean()))
print('MPJPE: ' + str(1000 * mpjpe.mean()))
print('PA-MPJPE: ' + str(1000 * pa_mpjpe.mean()))

print('G-V2V: ' + str(1000 * g_v2v.mean()))
print('V2V: ' + str(1000 * v2v.mean()))
print('PA-V2V: ' + str(1000 * pa_v2v.mean()))

# name = '_'.join(args.checkpoint.split("/")[-2:])
# save_path = "./eval_result/%s_all_loss.json"%name
# print("save_path: ", save_path)
# with open(save_path, 'w') as f:
#     tmp = {'v2v': v2v.tolist(), 'img_name': img_name_list}
#     json.dump(tmp, f, indent=4, sort_keys=True)




if args.save_results:
    pred_betas_list = torch.cat(pred_betas_list, dim=0).cpu().detach().numpy()
    pred_global_orient_list = torch.cat(pred_global_orient_list, dim=0).cpu().detach().numpy()
    pred_body_pose_list = torch.cat(pred_body_pose_list, dim=0).cpu().detach().numpy()
    pred_cam_full_list = torch.cat(pred_cam_full_list, dim=0).cpu().detach().numpy()

    model_id = args.checkpoint.split('/')[-2]
    output_res_folder = '{}/output_prohmr_baseline_{}_numsample_{}'.format(args.output_root, model_id, args.num_samples)
    os.mkdir(output_res_folder) if not os.path.exists(output_res_folder) else None
    output_res_dict = {}
    output_res_dict['pred_betas_list'] = pred_betas_list
    output_res_dict['pred_global_orient_list'] = pred_global_orient_list
    output_res_dict['pred_body_pose_list'] = pred_body_pose_list
    output_res_dict['pred_cam_full_list'] = pred_cam_full_list
    with open(os.path.join(output_res_folder, 'results_seed_{}.pkl'.format(args.seed)), 'wb') as result_file:
        pkl.dump(output_res_dict, result_file, protocol=2)
    print('[INFO] pred results saved to {}.'.format(output_res_folder))

    name = '_'.join(args.checkpoint.split("/")[-2:])
    save_path = "./eval_result/%s_results.json"%name
    print("save_path: ", save_path)

    #save the eval result
    result = {}
    result['g_mpjpe'] = str(1000 * g_mpjpe.mean())
    result['mpjpe'] = str(1000 * mpjpe.mean())
    result['pa_mpjpe'] = str(1000 * pa_mpjpe.mean())
    result['g_v2v'] = str(1000 * g_v2v.mean())
    result['v2v'] = str(1000 * v2v.mean())
    result['pa_v2v'] = str(1000 * pa_v2v.mean())
    result['args'] = args.__dict__
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)




