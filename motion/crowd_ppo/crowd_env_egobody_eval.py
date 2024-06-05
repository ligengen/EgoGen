"""
eval trained model with crowd motion
"""

import torch
import time
import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import pickle
import numpy as np
import sys 
sys.path.append(os.getcwd())
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from exp_GAMMAPrimitive.utils.batch_gen_amass import get_map
from crowd_ppo.utils import save_rollout_results
import pdb
import json
import warnings
import copy
# for rendering
import trimesh
from scipy.spatial.transform import Rotation
import pyrender
import smplx
import pytorch3d
import shapely
from shapely import LineString
from shapely.geometry import MultiPoint, Point, Polygon, mapping, MultiPolygon, LinearRing, mapping
from shapely import union_all
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
# import pyquaternion as pyquat
# from skimage.transform import resize


n_gens_2frame = 4 # smplx bug? error when bsize=1
# each env only produce one single data.
# all data in one batch are the same
# because smplx forward kinematics has bug when batch_size=1

class CrowdEnv(gym.Env):
    def __init__(self, init_env, save_rollout=True, render=False):
        self.rendering = render
        self.save_rollout = save_rollout

        # marker predictor and body regressor
        self.cfg, self.genop_2frame_male, self.genop_2frame_female, \
        bm_path, body_scene_data, self.smplxparser_1frame, self.smplxparser_2frame,\
        self.smplxparser_mp, self.feet_marker_idx, self.marker, self.vposer, self.man_id, self.cur_time = init_env

        device = torch.device('cuda', index=self.cfg.args.gpu_index)
        
        with torch.no_grad():
            self.data_mp0 = self._canonicalize_2frame(body_scene_data)
            self.body_scene_data = body_scene_data
            body_param_seed, prev_betas, gender, R0, T0 = self.data_mp0  # bsize=4
            nb, nt = body_param_seed.shape[:2]
            marker_seed = self.smplxparser_2frame.get_markers(
                betas=prev_betas,
                gender=gender,
                xb=body_param_seed.reshape(nb * nt, -1),
                to_numpy=False
            ).reshape(nb, nt, -1)
            marker_seed_l = marker_seed.reshape(marker_seed.shape[0], marker_seed.shape[1], -1, 3)
            marker_seed_w = torch.einsum('bij,btpj->btpi', R0, marker_seed_l) + T0[:,None,:,:]
            markers_w_xy = marker_seed_w[:, :, :, :2]
            # get body bbox on xy plane
            box_min = markers_w_xy.amin(dim=[1, 2]).reshape(n_gens_2frame, 1, 2)[0][0].cpu().numpy()
            box_max = markers_w_xy.amax(dim=[1, 2]).reshape(n_gens_2frame, 1, 2)[0][0].cpu().numpy()

        self.bbox = [[box_min[0], box_min[1]], [box_max[0], box_min[1]],\
                     [box_max[0], box_max[1]], [box_min[0], box_max[1]], [box_min[0], box_min[1]]]

        # the value is set in dummy_vector_env.py
        self.holes = [] # other human position

        # TODO: self defined range
        # gamma action: -5 ~ 6 
        self.action_space = Box(-6., 6., (128,))
        self.observation_space = Dict({"state": Box(-2., 2., (2, 402)), "egosensing": Box(-1., 1., (2, 32)),\
                                       "dist": Box(0., 1.), "time": Box(0., 1.)})

        if self.rendering:
            self.bm = smplx.create(bm_path, model_type='smplx',
                                gender='male', ext='npz',
                                num_pca_comps=12,
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                create_expression=True,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=True,
                                batch_size=2
                                ).eval().cuda()

    def step(self, action_z):
        # Run one timestep of environment dynamics.
        """check if agent has earned a reward (collision/goal/...) given action
            compute whether agent is done (collided/reached goal/ran out of time), the episode ends"""
        info = {}
        reward = 0.
        self.steps += 1
        if self.flag:
            print('the episode should be terminated! do not collect undefined states')
            pdb.set_trace()
        with torch.no_grad():
            # ==========================step once================================
            # self.state: [t,d]
            # a batch of same data. because of smplx bug
            states = self.state.reshape(self.state.shape[0], 1, self.state.shape[-1]).repeat(1,n_gens_2frame,1).detach() # t,b,d
            self.body_param_seed = self.body_param_seed.permute([1, 0, 2])
            t_his = states.shape[0]
            if t_his != 2:
                pdb.set_trace()
            t_pred = 20 - t_his
            X = states[:t_his]
            Xb = self.body_param_seed[:t_his]
            betas = self.betas.repeat(t_pred, X.shape[1], 1)

            if action_z.shape[0] == 128 and isinstance(action_z, np.ndarray): # only process batchsize=1!
                action_z = torch.from_numpy(action_z).cuda().unsqueeze(0).repeat(n_gens_2frame, 1)
            elif action_z.shape[0] == 128 and isinstance(action_z, torch.Tensor):
                action_z = action_z.unsqueeze(0).repeat(n_gens_2frame, 1)
            elif action_z.shape[0] == 1:
                action_z = action_z.repeat(n_gens_2frame, 1)

            Y_gen, Yb_gen = self.motion_model.model.sample_prior(X[:,:,:67*3], betas=betas, z=action_z)

            if torch.isnan(Y_gen).any() or torch.isinf(Y_gen).any():
                pdb.set_trace()
            if torch.isnan(Yb_gen).any() or torch.isinf(Yb_gen).any():
                pdb.set_trace()

            Y = torch.cat((X[:,:,:67*3], Y_gen), dim=0) # t=20,b=4,201
            Yb = torch.cat((Xb, Yb_gen), dim=0) # t=20,b=4,93

            # smooth the body poses, to eliminate the first-frame jump and compensate the regressor inaccuracy
            Yb = self._blend_params(Yb,t_his = t_his)

            pred_markers = Y.reshape(Y.shape[0], Y.shape[1], -1, 3).permute([1, 0, 2, 3])
            pred_params = Yb.permute([1, 0, 2])
            nb, nt = pred_params.shape[:2]
            if nt != 20:
                pdb.set_trace()

            # calc direction in state, local map, bparam_seed, and beta
            # pred_joints = self.smplxparser_mp.get_jts(betas=self.betas,
            #                                      gender=self.gender,
            #                                      xb=pred_params.reshape([nb*nt, -1]),
            #                                      to_numpy=False).reshape([nb, nt, -1, 3]) #[b=4,t=20,22,3]
            pred_output = self.smplxparser_mp.forward_smplx(betas=self.betas,
                                                            gender=self.gender,
                                                            xb=pred_params.reshape([nb*nt, -1]),
                                                            output_type='raw',
                                                            to_numpy=False)
            pred_joints_all = pred_output.joints[:,:,:].reshape([nb, nt, -1, 3]) #[b=4,t=20,J,3]
            # pred_joints_all = self.smplxparser_mp.get_all_jts(betas=self.betas,
            #                                      gender=self.gender,
            #                                      xb=pred_params.reshape([nb*nt, -1]),
            #                                      to_numpy=False).reshape([nb, nt, -1, 3])
            pred_joints = pred_joints_all[:, :, :22]
            pred_pelvis_loc = pred_joints[:, :, 0] # [b=4,t=20,3]
            # pred_markers_proj = self.smplxparser_mp.get_markers(betas=self.betas,
            #                                                gender=self.gender,
            #                                                xb=pred_params.reshape([nb*nt, -1]),
            #                                                to_numpy=False).reshape((nb, nt, -1, 3))
            pred_markers_proj = pred_output.vertices[:,self.marker,:].reshape((nb, nt, -1, 3))
            # b=4,t=20,67,3
            pred_marker_b = self.cfg.modelconfig.reproj_factor * pred_markers_proj + \
                            (1 - self.cfg.modelconfig.reproj_factor) * pred_markers

            if self.save_rollout:
                self.outmps.append([pred_marker_b, pred_params, self.betas, self.gender, self.R0[0], self.T0[0], pred_pelvis_loc, '2-frame'])

            # ========================= calc rewards start =========================
            # pred_params, pred_joints, pred_marker_b, R0, T0, wpath, points_local, local_map
            # bparams,     joints,      Y_l,           R0, T0, wpath, points_local, local_map

            # sdf penetration reward
            """
            pred_vertices = pred_output.vertices[:,:,:].reshape((nb, nt, -1, 3))
            vertices_w = torch.einsum("bij,btpj->btpi", self.R0, pred_vertices) + self.T0[:, None, :, :]
            remove feet vertices
            no_feet_v_w = vertices_w[:, :, self.no_feet_vids, :]
            sdf_values = calc_sdf(no_feet_v_w.reshape(nb * nt, -1, 3), self.scene_sdf).reshape(nb, -1) # [b, t*p]
            negative_values = sdf_values * (sdf_values < 0)
            r_pene = torch.exp((negative_values.sum(dim=-1) / nt / 512).clip(min=-100))
            num_inside = sdf_values.reshape(nb, nt, -1).lt(0.0).sum(dim=-1).max()
            self.pene_idx = torch.where(sdf_values.reshape(nb, nt, -1).lt(0.0)[0, -1, :])[0].detach().cpu().numpy()
            penetration = bool(num_inside > self.cfg.trainconfig.pene_thres)
            r_pene = torch.tensor(0.) if penetration else torch.tensor(0.05)
            """

            # foot skating reward
            h = 1 / 40
            Y_l_speed = torch.norm(pred_marker_b[:, 2:] - pred_marker_b[:, :-2], dim=-1) / 2. / h    # b=4,t=8,p=67
            dist2skat = (Y_l_speed[:, :, self.feet_marker_idx].amin(dim=-1) - 0.075).clamp(min=0).mean(dim=-1)
            r_skate = torch.exp(-dist2skat)[0] # remove batch dim

            # encourage move forward
            # r_nonstatic = (Y_l_speed[:, :, :].mean(dim=(1, 2)) > 0.1).float()[0]

            # filter out invalid data
            pred_joints_w = torch.einsum('bij,btpj->btpi', self.R0, pred_joints) + self.T0[:, None, :, :]
            pred_pelvis_loc_w = pred_joints_w[:, :, 0].detach().cpu().numpy()[0]
            for tmp_t in range(pred_pelvis_loc_w.shape[0]): 
                if not self.scene_poly.contains(Point(pred_pelvis_loc_w[tmp_t][:2])):
                    print("invalid pelvis location")
                    exit(-1)

            # floor penetration reward
            pred_marker_w = torch.einsum('bij,btpj->btpi', self.R0, pred_marker_b)\
                             + self.T0[:, None, :, :]  # [b, t, p, 3]
            dist2gp = torch.abs(pred_marker_w[:, :, self.feet_marker_idx, 2].amin(dim=-1) - 0.02).mean(dim=-1)
            r_floor = torch.exp(-dist2gp)[0]

            # vposer reward
            body_pose = pred_params[:, :, 6:69].reshape(nt*nb, -1)
            vp_embedding = self.vposer.encode(body_pose).loc
            latent_dim = vp_embedding.shape[-1]
            vp_norm = torch.norm(vp_embedding.reshape(nb, nt, -1), dim=-1).mean(dim=1)
            # terminate the episode if generated poses are unrealistic. thres to be tuned
            unrealistic_pose = bool((vp_norm > 13)[0])
            # change r_vp to r_pene like reward 
            r_vp = torch.tensor(0.) if unrealistic_pose else torch.tensor(0.05)
            if unrealistic_pose:
                print("unrealistic pose")
                exit(-1)

            # facing target reward
            joints_end = pred_joints[:, -1]  # [b,p,3]
            x_axis = joints_end[:, 2, :] - joints_end[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clip(min=1e-12)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=x_axis.device).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            b_ori = y_axis[:, :2]  # body forward dir of GAMMA is y axis

            target_wpath_l = torch.einsum('bij,btj->bti', self.R0.permute(0, 2, 1), \
                                                          self.body_scene_data['wpath'][-1:][None, ...] - self.T0)[:, :, :3]
            face_target_ori = target_wpath_l[:, 0, :2] - pred_pelvis_loc[:, -1, :2]
            face_target_ori = face_target_ori / torch.norm(face_target_ori, dim=-1, keepdim=True).clip(min=1e-12)
            r_face_target = ((torch.einsum('bi,bi->b', face_target_ori, b_ori) + 1) / 2.0)[0]

            # looking target reward
            pred_leye_pos = pred_joints_all[:, -1, 23]
            pred_reye_pos = pred_joints_all[:, -1, 24]
            eye_xaxis = pred_reye_pos - pred_leye_pos
            eye_xaxis[:, -1] = 0
            eye_xaxis = eye_xaxis / torch.norm(eye_xaxis, dim=-1, keepdim=True).clip(min=1e-12)
            eye_yaxis = torch.cross(z_axis, eye_xaxis)
            look_at = eye_yaxis[:, :2]
            r_look_target = ((torch.einsum('bi,bi->b', face_target_ori, look_at) + 1) / 2.0)[0]

            # reaching target reward
            dist2target = torch.norm(target_wpath_l[:, :, :] - pred_pelvis_loc[:, :, :], dim=-1, keepdim=False).clip(min=1e-12)[:, -1]
            r_target_dist = (self.dist - dist2target)[0]
            self.dist = dist2target
            r_goal = (self.dist < self.cfg.trainconfig.goal_thresh).float()[0]

            # set new state/obs. recanonicalize
            self.body_param_seed = pred_params[:,-t_his:] #[b=4,t=2,d]

            # move frame to the second last body's pelvis
            R_, T_ = self.smplxparser_1frame.get_new_coordinate(
                betas=self.betas,
                gender=self.gender,
                xb=self.body_param_seed[:,0],
                to_numpy=False)

            self.T0 = torch.einsum('bij,btj->bti', self.R0, T_) + self.T0
            self.R0 = torch.einsum('bij,bjk->bik', self.R0, R_)

            self.body_param_seed = self.smplxparser_2frame.update_transl_glorot(
                R_.repeat(t_his, 1, 1),
                T_.repeat(t_his, 1, 1),
                betas=self.betas,
                gender=self.gender,
                xb=self.body_param_seed.reshape(nb * t_his, -1),
                to_numpy=False,
                inplace=False).reshape(nb, t_his, -1) 

            marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0, 2, 1), pred_marker_b[:, -t_his:] - T_[..., None, :])
            pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0, 2, 1), pred_pelvis_loc[:, -t_his:] - T_)
            distxy, dist, fea_wpath, fea_marker, fea_marker_h, points_local, local_map = self._get_feature(marker_seed, pel_loc_seed, \
                                                                                    self.R0, self.T0, \
                                                                                    self.body_scene_data['wpath'][-1:],\
                                                                                    self.body_scene_data, self.holes)

            marker_seed = marker_seed.reshape(nb, t_his, -1) # [b,t,201]
            self.state = torch.cat([marker_seed, fea_marker], dim=-1) if 'condi' in self.cfg.modelconfig['body_repr'] else marker_seed

            # coap penetration detection
            # scene_pts_l = torch.einsum("bij,bpj->bpi", self.R0.permute(0, 2, 1), self.scene_vertices - self.T0)
            # TODO: how does face normals change after transformed to local?
            # collision_scene_points = self.smplxparser_1frame.get_collision_scene_pts(betas=self.betas,
            #                                                         gender=self.gender,
            #                                                         xb=self.body_param_seed[:, -1],
            #                                                         scene_vertices=scene_pts_l)
            # penetration = False
            # if self.rendering:
            #     self.bbox = None
            #     self.nocollidingpts = None
            #     self.collidingpts = None
            # if collision_scene_points is None:
            #     # no collision at all
            #     r_pene = torch.tensor(0.05)
            # else:
            #     r_pene = torch.tensor(0.)
            #     penetration = True
            #     # print("penetration!")
            #     # print(len(collision_scene_points))
            #     if self.rendering:
            #         collision_scene_points = torch.einsum("bij,bpj->bpi", self.R0, collision_scene_points.unsqueeze(0)) + self.T0
            #         self.collision_pts = collision_scene_points[0].detach().cpu().numpy()
            #         # self.collidingpts = torch.einsum("bij,bpj->bpi", self.R0, debug_info["points"]) + self.T0

            # 2D penetration detection
            marker_seed = marker_seed.reshape(marker_seed.shape[0], marker_seed.shape[1], -1, 3) 
            if self.cfg.lossconfig.pene_type == 'foot':
                markers_local_xy = marker_seed[:, :, self.feet_marker_idx, :2]  # [b, t, p, 2]
            elif self.cfg.lossconfig.pene_type == 'body':
                markers_local_xy = marker_seed[:, :, :, :2]
            # get body bbox on xy plane
            box_min = markers_local_xy.amin(dim=[1, 2]).reshape(nb, 1, 2)
            box_max = markers_local_xy.amax(dim=[1, 2]).reshape(nb, 1, 2)
            inside_feet_box = ((points_local[:, :, :2] >= box_min).all(-1) & \
                                (points_local[:, :, :2] <= box_max).all(-1)).float() #b,D
            num_pene = inside_feet_box * (1 - local_map) * 0.5# [b, D]
            self.pene_idx = num_pene[0]
            num_pene = num_pene.sum(dim=[1]) # [b]
            penetration = bool(num_pene[0] > self.cfg.trainconfig.pene_thres)
            # positive pene reward
            r_pene = torch.tensor(0.) if penetration else torch.tensor(0.05)

            reward = r_skate * self.cfg.lossconfig.weight_skate + \
                     r_floor * self.cfg.lossconfig.weight_floor + \
                     r_face_target * self.cfg.lossconfig.weight_face_target + \
                     r_look_target * self.cfg.lossconfig.weight_look_target + \
                     r_goal * self.cfg.lossconfig.weight_success + \
                     r_target_dist * self.cfg.lossconfig.weight_target_dist + \
                     r_pene * self.cfg.lossconfig.weight_pene + \
                     r_vp * self.cfg.lossconfig.weight_vp # + \
                     # r_nonstatic * self.cfg.lossconfig.weight_nonstatic
                     # r_move_toward * self.cfg.lossconfig.weight_move_toward + \
            for idx, r_value in enumerate((r_skate, r_floor, r_face_target, r_look_target, r_goal, r_target_dist, r_pene, r_vp)):#, r_nonstatic)):
                if torch.isnan(r_value).any() or torch.isinf(r_value).any():
                    pdb.set_trace()

            # update bbox for each agent for the global map
            marker_seed_w = torch.einsum('bij,btpj->btpi', self.R0, marker_seed) + self.T0[:,None,:,:]
            markers_w_xy = marker_seed_w[:, :, :, :2]
            # get body bbox on xy plane
            box_min = markers_w_xy.amin(dim=[1, 2]).reshape(n_gens_2frame, 1, 2)[0][0].cpu().numpy()
            box_max = markers_w_xy.amax(dim=[1, 2]).reshape(n_gens_2frame, 1, 2)[0][0].cpu().numpy()
            self.bbox = [[box_min[0], box_min[1]], [box_max[0], box_min[1]],\
                         [box_max[0], box_max[1]], [box_min[0], box_max[1]], [box_min[0], box_min[1]]]

            # remove batch dim
            self.state = self.state[0]
            # local_map = self.local_map[0]

            pred_joints_all = self.smplxparser_2frame.get_all_jts(betas=self.betas,
                                                    gender=self.gender,
                                                    xb=self.body_param_seed.reshape(nb * t_his, -1),
                                                    to_numpy=False).reshape(nb, t_his, -1, 3)
            # use global joints to render ego imgs
            pred_joints_all_w = torch.einsum('bij,btpj->btpi', self.R0, pred_joints_all) + self.T0[:, None, :, :]
            self.egosensing = self._calc_egosensing(pred_joints_all_w[0])
            
            # set terminated signal to end episode to avoid undefined states
            terminated = bool(r_goal > 0 or self.steps == self.max_depth)
            truncated = False 

            if self.save_rollout:
                if terminated or truncated:
                    self.flag = True
                    # if r_goal > 0:
                    save_rollout_results(self.body_scene_data, self.outmps, "./egobody_tmp_res/", self.man_id)

            next_obs = {"state": self.state, "egosensing": self.egosensing, \
                        "dist": (1/(dist2target+1))[0].reshape(1,), "time": torch.tensor(1 - self.steps / self.max_depth).cuda().reshape(1,)}

        if r_goal > 0:
            print('goal reached')
        
        return next_obs, reward.item(), terminated, truncated, info

    def reset(self):
        self.flag = False
        self.steps = 0
        if self.rendering:
            self.pene_idx = [0 for i in range(256)]
        with torch.no_grad():
            self.outmps = []
            self.scene_poly = self.body_scene_data['shapely_poly']
            # body_scene_data = self.scene_sampler.next_body(use_zero_pose=False,\
            #                                                random_rotation_range=(self.cfg.trainconfig.random_rotation_range))
            # data_mp0 = self._canonicalize_2frame(body_scene_data)
            # self.body_scene_data = body_scene_data
            # assume 20 frames corresponds to 0.6 m. additional 6m (used to be 3m) for exploration
            # self.max_depth = int(np.ceil((body_scene_data['path_len'] * 20  / .6 - 2) / 18)) + 3
            # self.max_depth = int(np.ceil(((body_scene_data['path_len'] + 6) * 20  / .6 - 2) / 18))
            self.max_depth = self.cfg.trainconfig.max_depth
            wpath = self.body_scene_data['wpath']
            body_param_seed, prev_betas, gender, R0, T0 = self.data_mp0  # bsize=4
            nb, nt = body_param_seed.shape[:2]
            if nb != n_gens_2frame:
                pdb.set_trace()
            t_his = 2

            marker_seed = self.smplxparser_2frame.get_markers(
                betas=prev_betas,
                gender=gender,
                xb=body_param_seed.reshape(nb * nt, -1),
                to_numpy=False
            ).reshape(nb, nt, -1)

            pelvis_loc = self.smplxparser_2frame.get_jts(betas=prev_betas,
                                                    gender=gender,
                                                    xb=body_param_seed.reshape(nb * nt, -1),
                                                    to_numpy=False
                                                    )[:, 0]  # [b*t, 3]
            pelvis_loc = pelvis_loc.reshape(nb, nt, -1)

            distxy, dist, fea_wpath, fea_marker, fea_marker_h, points_local, local_map = self._get_feature(marker_seed, pelvis_loc,
                                                                                         R0, T0, wpath[-1:], self.body_scene_data, self.holes)

            pred_joints_all = self.smplxparser_2frame.get_all_jts(betas=prev_betas,
                                                    gender=gender,
                                                    xb=body_param_seed.reshape(nb * nt, -1),
                                                    to_numpy=False).reshape(nb, nt, -1, 3)
            # use global joints to render ego imgs
            pred_joints_all_w = torch.einsum('bij,btpj->btpi', R0, pred_joints_all) + T0[:, None, :, :]
            self.egosensing = self._calc_egosensing(pred_joints_all_w[0])

            # env pretrained marker predictor network
            self.motion_model = self.genop_2frame_male if gender == 'male' else self.genop_2frame_female
            self.motion_model.model.predictor.eval()
            self.motion_model.model.regressor.eval()
                
            # body marker and direction
            self.state = torch.cat([marker_seed, fea_marker], dim=-1) if 'condi' in self.cfg.modelconfig['body_repr'] else marker_seed

            # remove batch dim, generate single state
            self.state = self.state[0]  #[1,402]
            # self.local_map = local_map[0]  #[256]
            self.betas = prev_betas  #[10]
            self.gender = gender  

            # do not remove batch dim. save changes for each step
            self.body_param_seed = body_param_seed
            self.R0 = R0
            self.T0 = T0

            self.dist = dist[:,0,0]

        return {"state": self.state, "egosensing": self.egosensing, \
                "dist": (1/(dist+1))[0][0], "time": torch.tensor(1 - self.steps / self.max_depth).cuda().reshape(1,)}, {} # info: ignore. don't delete it

    def render(self):
        self.body_scene_data['betas'] = self.body_scene_data['betas'].reshape(1,10)

        if self.rendering:
            # For render
            # need to update smplx bparam each step
            rotation = self.R0[0].reshape((3, 3))
            transl = self.T0[0].reshape((1, 3))
            pelvis_original = self.bm(betas=self.betas.unsqueeze(0)).joints[:, 0]
            self.body_scene_data['transl'] = torch.matmul((self.body_param_seed[0, :, :3] + pelvis_original),
                                                          rotation.T) - pelvis_original + transl
            r_ori = Rotation.from_rotvec(self.body_param_seed[0, :, 3:6].cpu())
            r_new = Rotation.from_matrix(rotation.cpu()) * r_ori
            self.body_scene_data['global_orient'] = torch.from_numpy(r_new.as_rotvec()).type(torch.float32).cuda()
            self.body_scene_data['body_pose'] = self.body_param_seed[0, :, 6:69]

        init_body_mesh = trimesh.Trimesh(
            vertices=self.bm(**self.body_scene_data).vertices[0].detach().cpu().numpy(),
            faces=self.bm.faces,
            vertex_colors=np.array([100, 200, 100])
        )
        init_body_mesh2 = trimesh.Trimesh(
            vertices=self.bm(**self.body_scene_data).vertices[1].detach().cpu().numpy(),
            faces=self.bm.faces,
            vertex_colors=np.array([100, 100, 200])
        )
        
        obj_mesh = trimesh.load(self.body_scene_data['scene_path'], force='mesh')
        obj_mesh.vertices[:, 2] -= 0.02
        vis_mesh = [
            init_body_mesh,
            init_body_mesh2,
            # obj_mesh, 
            self.body_scene_data['navmesh'],
            # trimesh.creation.axis()
            ]

        for point_idx, pelvis in enumerate(self.body_scene_data['wpath'].reshape(-1, 3)):
            trans_mat = np.eye(4)
            trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
            point_axis = trimesh.creation.axis(transform=trans_mat)
            vis_mesh.append(point_axis)

        # collision_nodes = []
        # if self.collision_pts is not None:
        #     for pid, collision_pt in enumerate(self.collision_pts):
        #         sm = trimesh.creation.uv_sphere(radius=0.03)
        #         sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        #         tfs = np.tile(np.eye(4), (1, 1, 1))
        #         tfs[:, :3, 3] = collision_pt
        #         m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        #         collision_nodes.append(pyrender.Node(mesh=m, name='collision'))


        # visualize map
        joints = self.bm(**self.body_scene_data).joints  # [b,p,3]

        ray_end_nodes = []
        endpts = np.zeros((32, 3))
        endpts[:, :2] = self.end_pts[0]
        endpts[:, 2] = joints[0,23][2].item()
        for i in range(32):
            sm = trimesh.creation.uv_sphere(radius=0.02)
            sm.visual.vertex_colors = [0.0, 1.0, 0.0]
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[:, :3, 3] = endpts[i]
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            ray_node = pyrender.Node(mesh=m, name='f1_sensing')
            ray_end_nodes.append(ray_node)

        endpts = np.zeros((32, 3))
        endpts[:, :2] = self.end_pts[1]
        endpts[:, 2] = joints[1,23][2].item()
        for i in range(32):
            sm = trimesh.creation.uv_sphere(radius=0.02)
            sm.visual.vertex_colors = [0.0, 0.0, 1.0]
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[:, :3, 3] = endpts[i]
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            ray_node = pyrender.Node(mesh=m, name='f2_sensing')
            ray_end_nodes.append(ray_node)

        # visualize debug info
        # if self.bbox is not None:
            # vis_mesh.append(trimesh.creation.box(bounds=self.bbox))
        #     box = trimesh.creation.box(bounds=[self.bbox['bb_min'][0].detach().cpu().numpy(), self.bbox['bb_max'][0].detach().cpu().numpy()])
            # box.visual.face_colors = [1., 1., 1., 0.1]
        #     vis_mesh.append(box)
        #     print(self.box)

        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
        points_local, points, map = get_map(self.body_scene_data['navmesh'], gamma_orient, gamma_transl,
                                res=self.cfg.modelconfig.map_res, extent=self.cfg.modelconfig.map_extent,
                                return_type='numpy')
        points = points[0]  # [p, 3]
        map = map[0]  #[p]
        cells = []
        for point_idx in range(points.shape[0]):
            color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 100])
            if self.pene_idx[point_idx] != 0:
            # if point_idx in self.pene_idx:
                color = np.array([0,0,0,200])
            transform = np.eye(4)
            transform[:3, 3] = points[point_idx]
            cell = trimesh.creation.box(extents=(0.05, 0.05, 1), vertex_colors=color, transform=transform)
            cells.append(cell)
        vis_mesh.append(trimesh.util.concatenate(cells))

        # print(self.body_scene_data['wpath'])
        scene = pyrender.Scene()
        for mesh in vis_mesh:
            scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
        for node in ray_end_nodes:
            scene.add_node(node)
        # for node in collision_nodes:
        #     scene.add_node(node)
        pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

    def seed(self, seed):
        # enable parallel sampling 
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _calc_egosensing(self, joint):
        # future: calc with torch tensor
        # joint: [t=2, J=127, 3]

        # 57: leye 56: reye
        joint = joint.detach().cpu().numpy()
        look_at = joint[:, 57] - joint[:, 23] + joint[:, 56] - joint[:, 24]
        look_at = look_at.astype(np.float64)
        look_at[:, -1] = 0.
        look_at = look_at / np.linalg.norm(look_at, axis=-1, keepdims=True) # look at dir in 2d
        eye_2d = (joint[:, 23] + joint[:, 24]) / 2
        eye_2d[:, -1] = 0.
        # shoot 32 rays (ray_len: 7) to calc nearest obstacle 
        angle_grids = np.linspace(-np.pi/2, np.pi/2, 32)
        rays = np.zeros((2, 2, 32)) # [t, 2, 32]
        rays[:, 0, :] = np.cos(angle_grids)
        rays[:, 1, :] = np.sin(angle_grids)
        # rays = np.array([[look_at[0], -look_at[1]], [look_at[1], look_at[0]]]) @ rays
        rot = np.asarray([[[look_at[0][0], -look_at[0][1]], [look_at[0][1], look_at[0][0]]], \
                        [[look_at[1][0], -look_at[1][1]], [look_at[1][1], look_at[1][0]]]])
        rays = np.einsum("tnk,tkm->tnm", rot, rays)
        rays = np.swapaxes(rays, 1, 2)

        obstacle_dists = []
        # lines = MultiLineString([i for i in zip(eye_2d[:2][np.newaxis, :].repeat(32, 0), rays * 1 + eye_2d[:2])])
        # intersects = lines.intersection(self.scene_poly)
        # for cropped_ray in intersects.geoms:
        #     if cropped_ray.distance(start_p) < 1e-4:
        #         self.end_pts.append(cropped_ray.coords[1])

        start_p = MultiPoint(eye_2d[:, :2]) # 2-frame eye loc
        ray_len = 7
        if self.rendering:
            ray_len = 2
        self.end_pts = [[], []] #[t=2, 32, 2]
        eye_in_scene = [True, True]
        # if self.rendering:
        #     ray_len = 2

        # first check if the starting points are in the scene
        for idx, sp in enumerate(start_p.geoms):
            # 1-f eye loc: sp 
            if not self.scene_poly_dyn.contains(sp):
                self.end_pts[idx] = eye_2d[idx, :2][np.newaxis, :].repeat(32, axis=0)
                eye_in_scene[idx] = False
                # return torch.cuda.FloatTensor(np.linalg.norm(np.array(self.end_pts) - eye_2d[:, :2], axis=1) / ray_len)
                # return torch.zeros((2,32)).cuda()

        for idx, start_p_ in enumerate(start_p.geoms):
            if not eye_in_scene[idx]:
                continue

            for i in range(32):
                line = LineString([eye_2d[idx, :2], rays[idx, i] * ray_len + eye_2d[idx, :2]])    
                with warnings.catch_warnings(record=True) as w:
                    intersect = line.intersection(self.scene_poly_dyn)
                    if len(w) > 0: 
                        now = time.time()
                        with open('line_%s.pkl' % now, 'wb') as g:
                            pickle.dump(line, g)
                        with open('scene_poly_%s.pkl' % now, 'wb') as f:
                            pickle.dump(self.scene_poly_dyn, f)
                        np.save("rays_%s.npy" % now, rays)
                        print('saved!!')
                if intersect.geom_type == "MultiLineString":
                    for geom in intersect.geoms:
                        if geom.geom_type == "LineString":
                            if geom.distance(start_p_) < 1e-5:
                                self.end_pts[idx].append(geom.coords[1])
                                break
                        elif geom.geom_type == "Point":
                            if geom.distance(start_p_) < 1e-5:
                                self.end_pts[idx].append(geom.coords[0])
                                break
                        else:
                            print(geom.geom_type)
                            pdb.set_trace()
                elif intersect.geom_type == "LineString":
                    if intersect.distance(start_p_) < 1e-5:
                        self.end_pts[idx].append(intersect.coords[1])
                else:
                    print(intersect.geom_type)
                    pdb.set_trace()
            if len(self.end_pts[idx]) != 32:
                # if len(self.end_pts[idx]) == 0:
                #     # eye out of the scene. can not calc rays
                #     self.end_pts = [eye_2d[:2] for i in range(32)]
                # else:
                pdb.set_trace()
        # scale [0, 1] to [-1, 1]
        return -1 + 2 * torch.cuda.FloatTensor(np.linalg.norm(np.array(self.end_pts) - eye_2d[:, np.newaxis, :2], axis=-1) / ray_len)

    def _canonicalize_2frame(self, data):
        t_his = 2
        smplx_transl = data['motion_seed']['transl']
        smplx_glorot = data['motion_seed']['global_orient']
        smplx_poses = data['motion_seed']['body_pose']
        gender = data['gender']
        smplx_handposes = torch.cuda.FloatTensor(smplx_transl.shape[0], 24).zero_()
        prev_params = torch.cat([smplx_transl, smplx_glorot,
                                 smplx_poses, smplx_handposes],
                                dim=-1)  # [t,d]
        prev_params = prev_params.repeat(n_gens_2frame, 1, 1)
        prev_betas = data['betas']
        nb, nt = prev_params.shape[:2]
        ## move frame to the body's pelvis
        R0, T0 = self.smplxparser_1frame.get_new_coordinate(
            betas=prev_betas,
            gender=gender,
            xb=prev_params[:, 0],
            to_numpy=False)

        ## get the last body param and marker in the new coordinate
        body_param_seed = self.smplxparser_2frame.update_transl_glorot(R0.repeat(t_his, 1, 1), T0.repeat(t_his, 1, 1),
                                                                  betas=prev_betas,
                                                                  gender=gender,
                                                                  xb=prev_params.reshape(nb * nt, -1),
                                                                  to_numpy=False,
                                                                  inplace=False
                                                                  ).reshape(nb, nt, -1)

        return body_param_seed, prev_betas, gender, R0, T0

    def _canonicalize_static_pose(self, data):
        smplx_transl = data['transl']
        smplx_glorot = data['global_orient']
        smplx_poses = data['body_pose']
        gender = data['gender']
        smplx_handposes = torch.cuda.FloatTensor(smplx_transl.shape[0], 24).zero_()
        prev_params = torch.cat([smplx_transl, smplx_glorot,
                                 smplx_poses, smplx_handposes],
                                dim=-1)  # [t,d]
        # smplx forward bug. bsize can not = 1
        prev_params = prev_params.repeat(n_gens_2frame, 1, 1)
        prev_betas = data['betas']
        nb, nt = prev_params.shape[:2]

        ## move frame to the body's pelvis
        # ? batch data and single data can have different joints
        R0, T0 = self.smplxparser_1frame.get_new_coordinate(
            betas=prev_betas,
            gender=gender,
            xb=prev_params[:, 0],
            # xb=prev_params,
            to_numpy=False)

        ## get the last body param and marker in the new coordinate
        body_param_seed = self.smplxparser_1frame.update_transl_glorot(R0, T0,
                                                                  betas=prev_betas,
                                                                  gender=gender,
                                                                  xb=prev_params.reshape(nb * nt, -1),
                                                                  # xb=prev_params.reshape(nt, -1),
                                                                  to_numpy=False
                                                                  ).reshape(nb, nt, -1)
                                                                  # ).reshape(1, nt, -1)
        return body_param_seed, prev_betas, gender, R0, T0

    def _get_dynamic_map(self, R, T, polygon):
        res = self.cfg.modelconfig.map_res
        extent = self.cfg.modelconfig.map_extent
        batch_size = R.shape[0]
        x = torch.linspace(-extent, extent, res)
        y = torch.linspace(-extent, extent, res)
        xv, yv = torch.meshgrid(x, y)
        points = torch.stack([xv, yv, torch.zeros_like(xv)], axis=2).to(device='cuda')  # [res, res, 3]
        points = points.reshape(1, -1, 3).repeat(batch_size, 1, 1)
        points_scene = torch.einsum('bij,bpj->bpi', R, points) + T  # [b, r*r, 3]
        # TODO: may need to change if the floor height is not 0
        points_scene[:, :, 2] = 0.
        points_2d = points_scene[:, :, :2].reshape(batch_size * res * res, 2).detach().cpu().numpy()  # [P=b*r*r, 2]
        map = []
        for pt in points_2d:
            try:
                map.append(polygon.contains(Point(pt)))
            except:
                shapely.plotting.plot_polygon(polygon)
                plt.show()
        map = torch.cuda.FloatTensor(map).reshape(batch_size, res * res)
        
        return points, map

    def _get_feature(self, Y_l, pel, R0, T0, pt_wpath, scene, holes):
        '''
        --Y_l = [b,t,d] local marker
        --pel = [b,t,d]
        --pt_wpath = [1,d]
        '''
        nb, nt = pel.shape[:2]
        Y_l = Y_l.reshape(nb, nt, -1, 3)
        pt_wpath_l_3d = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)

        '''extract path feature = normalized direction + unnormalized height'''
        fea_wpathxy = pt_wpath_l_3d[:, :, :2] - pel[:, :, :2]
        fea_wpathxyz = pt_wpath_l_3d[:, :, :3] - pel[:, :, :3]
        dist_xy = torch.norm(fea_wpathxy, dim=-1, keepdim=True).clip(min=1e-12)
        dist_xyz = torch.norm(fea_wpathxyz, dim=-1, keepdim=True).clip(min=1e-12)
        fea_wpathxy = fea_wpathxy / dist_xy
        fea_wpathz = pt_wpath_l_3d[:, :, -1:] - pel[:, :, -1:]
        fea_wpath = torch.cat([fea_wpathxy, fea_wpathz], dim=-1)

        '''extract marker feature'''
        fea_marker = pt_wpath_l_3d[:, :, None, :] - Y_l
        dist_m_3d = torch.norm(fea_marker, dim=-1, keepdim=True).clip(min=1e-12)
        fea_marker_3d_n = (fea_marker / dist_m_3d).reshape(nb, nt, -1)

        '''extract marker feature with depth'''
        dist_m_2d = torch.norm(fea_marker[:, :, :, :2], dim=-1, keepdim=True).clip(min=1e-12)
        fea_marker_xyn = fea_marker[:, :, :, :2] / dist_m_2d
        fea_marker_h = torch.cat([fea_marker_xyn, fea_marker[:, :, :, -1:]], dim=-1).reshape(nb, nt, -1)

        """local map"""
        # exterior = MultiPoint(scene['navmesh'].vertices[:,:2]).convex_hull

        # valid_region = np.asarray(mapping(exterior)['coordinates'][0]) # clock-wise 
        # holes = torch.cuda.FloatTensor(self.holes)

        holes = np.array(holes)
        holes_poly = [Polygon(hole) for hole in holes]
        holes_multipoly = union_all(holes_poly)
        holes = []
        if type(holes_multipoly)==shapely.geometry.multipolygon.MultiPolygon:
            pdb.set_trace()
            for poly in holes_multipoly.geoms:
                holes.append(poly.exterior.coords[::-1][::-1])
        elif type(holes_multipoly)==shapely.geometry.polygon.Polygon:
            holes.append(holes_multipoly.exterior.coords[:])
        else:
            pdb.set_trace()
        polygon = Polygon(self.scene_poly, holes)
        points_local, map = self._get_dynamic_map(R0, T0, polygon) 
        # calc egosensing using updated scene polygon
        self.scene_poly_dyn = polygon

        local_map = map.float()  # [b, res*res]
        local_map[map == False] = -1  #  reassign 0(non-walkable) to -1, https://stats.stackexchange.com/a/138370
        # local_map = torch.zeros_like(local_map)  # test dump map
        # print(time.time() - t1)

        for idx, feature in enumerate((dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h)):# , points_local, local_map)):
            if torch.isnan(feature).any() or torch.isinf(feature).any():
                print('feature ', idx, 'is not valid')
                print(feature)
                pdb.set_trace()

        return dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h, points_local, local_map

    def _blend_params(self, body_params, t_his):
        start_idx = 6
        param_n = body_params[t_his-1, :, start_idx:]
        param_p = body_params[t_his+1, :, start_idx:]
        body_params[t_his, :, start_idx:] = (param_n+param_p)/2.0

        t_his = t_his+1
        param_n = body_params[t_his-1, :, start_idx:]
        param_p = body_params[t_his+1, :, start_idx:]
        body_params[t_his, :, start_idx:] = (param_n+param_p)/2.0
        return body_params


if __name__ == "__main__":
    from crowd_ppo.primitive_model import load_model
    from human_body_prior.tools.model_loader import load_vposer
    from exp_GAMMAPrimitive.utils.environments import BatchGeneratorScene2frameTrain as SceneTrainGen
    from pathlib import Path
    from exp_GAMMAPrimitive.utils import config_env
    from models.baseops import SMPLXParser
    import json

    # init env 
    cfg, \
    genop_2frame_male, \
    genop_2frame_female = load_model()
    # dataset
    scene_list_path = Path('data/scenes/random_box_obstacle_new_names.pkl')
    with open(scene_list_path, 'rb') as f:
        scene_list = pickle.load(f)
    print('#scene: ', len(scene_list))
    bm_path = config_env.get_body_model_path()

    vposer, _ = load_vposer(bm_path + '/vposer_v1_0', vp_model='snapshot')
    vposer.eval()
    vposer.to('cuda')

    motion_seed_list = list(Path("data/locomotion").glob("*npz"))
    # TODO: should rewrite batch_gen_amass.BatchGeneratorReachingTarget not load smplx and vposer everytime
    scene_sampler = SceneTrainGen(dataset_path='', motion_seed_list=motion_seed_list,
                                  # scene_dir='/mnt/scratch/kaizhao/datasets/replica/room_0', # scene_list=scene_list,
                                  scene_dir='exp_data/scenes', scene_list=scene_list,
                                  # scene_type='room_0', body_model_path=bm_path) 
                                  scene_type='random_box_obstacle_new', body_model_path=bm_path) 

    # scene_sdf = np.load("data/room0_sdf.pkl", allow_pickle=True)
    # for key, value in scene_sdf.items():
    #     scene_sdf[key] = torch.as_tensor(value, dtype=torch.float32).cuda()

    # smplx parser
    pconfig_1frame = {
        'n_batch': 1 * n_gens_2frame,
        'device': 'cuda',
        'marker_placement': 'ssm2_67'
    }
    smplxparser_1frame = SMPLXParser(pconfig_1frame)

    pconfig_2frame = {
        'n_batch': 2 * n_gens_2frame,
        'device': 'cuda',
        'marker_placement': 'ssm2_67'
    }
    smplxparser_2frame = SMPLXParser(pconfig_2frame)

    pconfig_mp = {
        'n_batch': 20 * n_gens_2frame,
        'device': 'cuda',
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp = SMPLXParser(pconfig_mp)

    with open(config_env.get_body_marker_path() + '/SSM2.json') as f:
        marker_ssm_67 = json.load(f)['markersets'][0]['indices']
    feet_markers = ['RHEE', 'RTOE', 'RRSTBEEF', 'LHEE', 'LTOE', 'LRSTBEEF']
    feet_marker_idx = [list(marker_ssm_67.keys()).index(marker_name) for marker_name in feet_markers]
    body_markers = list(marker_ssm_67.values())

    init_env = [cfg, genop_2frame_male, genop_2frame_female, \
                bm_path, scene_sampler, smplxparser_1frame, smplxparser_2frame,\
                smplxparser_mp, feet_marker_idx, body_markers, vposer]#, scene_sdf]

    env = CrowdEnv(init_env, render=True)
    env.reset()
    env.render()
    for _ in range(100):
        obs,rew,t,t2,_ = env.step(torch.zeros(1,128).cuda())
        env.render()

