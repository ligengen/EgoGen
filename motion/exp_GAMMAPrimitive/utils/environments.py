import os, sys
import pickle
import random

import numpy as np
import pytorch3d.transforms
import torch
import trimesh
from shapely.geometry import Polygon, Point, MultiPoint, mapping
from shapely import union_all
import shapely

sys.path.append(os.getcwd())

from models.baseops import SMPLXParser
from exp_GAMMAPrimitive.utils.batch_gen_amass import *

rest_pose = torch.cuda.FloatTensor(
[0.0, 0.0, 0.0, -0.011472027748823166, 1.2924634671010859e-26, 2.5473026963570487e-18, -0.0456559844315052, -0.0019564421381801367, -0.08563289791345596, 0.11526273936033249, 0.0, -2.5593469423841883e-17, 0.06192377582192421, -1.2932950836510723e-26, -1.3749840337845367e-17, 0.07195857912302017, 0.00617849500849843, 1.4564738304301272e-11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.09871290624141693, 6.4515848478496824e-18, 7.851343602724672e-17, -0.008369642309844494, -0.12677378952503204, -0.3995564579963684, 0.0013758527347818017, 0.01013219729065895, 0.23814785480499268, 0.277565598487854, -1.5771439149242302e-17, -6.061879960787066e-17, -0.10060133039951324, 0.1710081696510315, -0.8297445774078369, 0.016900330781936646, -0.03264763951301575, 0.9994331002235413, -0.11047029495239258, -0.4468419551849365, -0.17531509697437286, -0.15802216529846191, 0.4728464186191559, 0.023101171478629112, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11167845129966736, 0.04289207234978676, -0.41644084453582764, 0.10881128907203674, -0.06598565727472305, -0.756219744682312, -0.0963931530714035, -0.09091583639383316, -0.18845966458320618, -0.11809506267309189, 0.050943851470947266, -0.5295845866203308, -0.14369848370552063, 0.055241718888282776, -0.704857349395752, -0.019182899966835976, -0.0923367589712143, -0.3379131853580475, -0.45703303813934326, -0.1962839663028717, -0.6254575848579407, -0.21465237438678741, -0.06599827855825424, -0.5068942308425903, -0.36972442269325256, -0.0603446289896965, -0.07949023693799973, -0.14186954498291016, -0.08585254102945328, -0.6355276107788086, -0.3033415675163269, -0.05788097903132439, -0.6313892006874084, -0.17612087726593018, -0.13209305703639984, -0.3733545243740082, 0.850964367389679, 0.2769227623939514, -0.09154807031154633, -0.4998386800289154, 0.026556432247161865, 0.052880801260471344, 0.5355585217475891, 0.045960985124111176, -0.27735769748687744, 0.11167845129966736, -0.04289207234978676, 0.41644084453582764, 0.10881128907203674, 0.06598565727472305, 0.756219744682312, -0.0963931530714035, 0.09091583639383316, 0.18845966458320618, -0.11809506267309189, -0.050943851470947266, 0.5295845866203308, -0.14369848370552063, -0.055241718888282776, 0.704857349395752, -0.019182899966835976, 0.0923367589712143, 0.3379131853580475, -0.45703303813934326, 0.1962839663028717, 0.6254575848579407, -0.21465237438678741, 0.06599827855825424, 0.5068942308425903, -0.36972442269325256, 0.0603446289896965, 0.07949023693799973, -0.14186954498291016, 0.08585254102945328, 0.6355276107788086, -0.3033415675163269, 0.05788097903132439, 0.6313892006874084, -0.17612087726593018, 0.13209305703639984, 0.3733545243740082, 0.850964367389679, -0.2769227623939514, 0.09154807031154633, -0.4998386800289154, -0.026556432247161865, -0.052880801260471344, 0.5355585217475891, -0.045960985124111176, 0.27735769748687744]
)[3:66].reshape(1, 63)

class BatchGeneratorScene2frameTrain(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 motion_seed_list=None,
                 scene_list=None,
                 scene_dir=None,
                 scene_type='random',
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene_list= scene_list
        self.scene_dir = Path(scene_dir)
        self.scene_idx = 0
        self.scene_type=scene_type

        self.motion_seed_list = motion_seed_list
        self.bm_2frame = smplx.create(body_model_path, model_type='smplx',
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
        self.navmesh_path = self.scene_dir / 'navmesh_tight.ply'
        self.navmesh = trimesh.load(self.navmesh_path, force='mesh')
        samples_path = 'data/room0_samples.pkl'
        with open(samples_path, 'rb') as f:
            self.sample_pairs = pickle.load(f)
        shapely_path = 'data/replica_room0_shapely.pkl'
        self.shapely_poly = np.load(shapely_path, allow_pickle=True)
        motion_seed_path = "data/locomotion/subseq_00343.npz"
        self.motion_data = np.load(motion_seed_path)

    # TODO: change to random wpath gen!
    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                  scene_idx=None, start_target=None, random_rotation_range=1.0,
                  clip_far=False,
                  res=32, extent=1.6, last_motion=None):
        if scene_idx is None and self.scene_list is not None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        if self.scene_list is not None:
            scene_name = self.scene_list[scene_idx]
        if self.scene_type == 'prox':
            mesh_path = self.scene_dir / 'PROX' / (scene_name + '_floor.ply')
            navmesh_path = self.scene_dir / 'PROX' / (scene_name + '_navmesh.ply')
        elif self.scene_type == 'room_0':
            mesh_path = self.scene_dir / 'mesh_floor.ply'
            # samples_path = 'data/room0_samples_new_ori.pkl'
            # shapely_path = 'data/replica_room0_shapely.pkl'
        elif 'random' in self.scene_type or 'exploration' in self.scene_type:
            mesh_path = self.scene_dir / self.scene_type / (scene_name + '.ply')
            navmesh_path = self.scene_dir / self.scene_type / (scene_name + '_navmesh_tight.ply')
            # self.navmesh_path = navmesh_path
            samples_path = self.scene_dir / self.scene_type / (scene_name + '_samples.pkl')
            shapely_path = self.scene_dir / self.scene_type / (scene_name + '_shapely.pkl')
            # navmesh = trimesh.load(navmesh_path, force='mesh')
            # navmesh_loose_path = self.scene_dir / self.scene_type / (scene_name + '_navmesh.ply')
            # navmesh_loose = trimesh.load(navmesh_loose_path, force='mesh')
        navmesh = self.navmesh
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 50])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )
        shapely_poly = None
        # if os.path.exists(shapely_path):
        #     # shapely_poly = np.load(shapely_path, allow_pickle=True)
        shapely_poly = self.shapely_poly

        # import pyrender
        # scene = pyrender.Scene()
        # scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
        # # scene.add(pyrender.Mesh.from_trimesh(navmesh, smooth=False))
        # scene.add(pyrender.Mesh.from_trimesh(navmesh_crop, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        """randomly specify a 3D path"""
        wpath = np.zeros((3,3))
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        if start_target is not None:
            wpath[0] = torch.cuda.FloatTensor(start_target[0])  # starting point
            wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
        elif self.scene_type == 'prox':
            start_target = np.zeros((2, 3))  # pairs of start and target positions
            max_try = 32
            for try_idx in range(max_try):
                start_target[0] = trimesh.sample.sample_surface_even(navmesh, 1)[0]
                if try_idx < max_try - 1:
                    crop_box = trimesh.creation.box(extents=[sigma, sigma, 2])
                    crop_box.vertices += start_target[0]
                    navmesh_crop = navmesh.slice_plane(crop_box.facets_origin, -crop_box.facets_normal)
                    if len(navmesh_crop.vertices) >= 3:
                        start_target[1] = trimesh.sample.sample_surface_even(navmesh_crop, 1)[0]
                        break
                else:
                    start_target[1] = trimesh.sample.sample_surface_even(navmesh, 1)[0]

            if np.isnan(start_target).any():
                print('error in sampling start-target')
            wpath[0] = torch.cuda.FloatTensor(start_target[0]) #starting point
            wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
            if torch.isnan(wpath).any() or torch.isinf(wpath).any():
                print('error:wpath invalid, random sample', wpath)
        elif 'random' in self.scene_type or self.scene_type == 'room_0' or 'exploration' in self.scene_type:
            # with open(samples_path, 'rb') as f:
            #     sample_pairs = pickle.load(f)
            sample_pairs = self.sample_pairs
            num_samples = len(sample_pairs)
            if num_samples == 0:
                print('error: zero samples, precompute')

            # use path planning to know path len prior to determine the max_depth in RL
            # (start, target), path_len = sample_pairs[np.random.randint(low=0, high=num_samples)]
            start, target = sample_pairs[np.random.randint(low=0, high=num_samples)]
            # random wpath sampling 
            # start = np.array(trimesh.sample.sample_surface_even(navmesh_loose, 1)[0])
            # target = np.array(trimesh.sample.sample_surface_even(navmesh_loose, 1)[0])

            if clip_far and np.linalg.norm(target - start) > 1.0:
                # print('clip far pairs')
                length = np.linalg.norm(target - start).clip(min=1e-12)
                vec_dir = (target - start) / length
                l1 = np.random.uniform(low=0.0, high=length - 0.5)
                l2 = min(np.random.uniform(0.5, 1.0) + l1, length)
                target = start + vec_dir * l2
                start = start + vec_dir * l1
            wpath[0] = torch.cuda.FloatTensor(start)  # starting point
            wpath[1] = torch.cuda.FloatTensor(target)  # ending point xy
            # print("start: ", start)
            # print("target: ", target)
            if torch.isnan(wpath).any() or torch.isinf(wpath).any():
                print('error:wpath invalid, precompute', wpath)

        # wpath[2, :2] = wpath[0, :2] + torch.randn(2).to(device=wpath.device) #point to initialize the body orientation, not returned
        theta = torch.pi * (2 * torch.cuda.FloatTensor(1).uniform_() - 1) * random_rotation_range
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                                  convention="XYZ")
        wpath[2] = torch.einsum('ij, j->i', random_rotz[0], wpath[1] - wpath[0]) + wpath[0]  # face the target with [-90, 90] disturbance
        if torch.norm(wpath[2] - wpath[0], dim=-1) < 1e-12:
            wpath[2] += 1e-12
        # hard code
        # wpath[0] = torch.cuda.FloatTensor([-1, 0, 0])
        # wpath[1] = torch.cuda.FloatTensor([0.5, 0, 0])
        # wpath[2] = wpath[1]

        """generate init body"""
        # motion_seed_path = random.choice(self.motion_seed_list)
        
        # TODO: cleaner initial motion seed
        # motion_seed_path = "data/locomotion/subseq_00343.npz"
        # motion_data = np.load(motion_seed_path)
        if last_motion is None:
            motion_data = self.motion_data
            if len(motion_data['poses']) < 18:
                pdb.set_trace()
            # start_frame = torch.randint(0, len(motion_data['poses']) - 1, (1,)).item()
            start_frame = 5

            motion_seed_dict = {}
            motion_seed_dict['betas'] = torch.cuda.FloatTensor(motion_data['betas']).reshape((1, 10)).repeat(2, 1)
            motion_seed_dict['body_pose'] = torch.cuda.FloatTensor(motion_data['poses'][start_frame:start_frame + 2, 3:66])
            motion_seed_dict['global_orient'] = torch.cuda.FloatTensor(motion_data['poses'][start_frame:start_frame + 2, :3])
            motion_seed_dict['transl'] = torch.cuda.FloatTensor(motion_data['trans'][start_frame:start_frame + 2])
        else:
            motion_seed_dict = {}
            motion_seed_dict['betas'] = torch.cuda.FloatTensor(np.zeros((1, 10))).repeat(2, 1)
            motion_seed_dict['body_pose'] = torch.cuda.FloatTensor(last_motion[:, 6:69])
            motion_seed_dict['global_orient'] = torch.cuda.FloatTensor(last_motion[:, 3:6])
            motion_seed_dict['transl'] = torch.cuda.FloatTensor(last_motion[:, :3])

        # randomly rotate around up-vec
        """
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi * 2
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
                                                                 convention="XYZ").reshape(1, 3, 3)
        pelvis_zero = self.bm_2frame(betas=motion_seed_dict['betas']).joints[:1, 0, :]  # [1, 3]
        original_rot = pytorch3d.transforms.axis_angle_to_matrix(motion_seed_dict['global_orient'])
        new_rot = torch.einsum('bij,bjk->bik', random_rot, original_rot)
        new_transl = torch.einsum('bij,bj->bi', random_rot, pelvis_zero + motion_seed_dict['transl']) - pelvis_zero
        motion_seed_dict['global_orient'] = pytorch3d.transforms.matrix_to_axis_angle(new_rot)
        motion_seed_dict['transl'] = new_transl
        """
        if last_motion is None:
            # rotate the body to face the target
            joints = self.bm_2frame(**motion_seed_dict).joints
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clip(min=1e-12)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=x_axis.device).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            b_ori = y_axis[0]
            b_ori /= torch.linalg.norm(b_ori)    
            target_ori = wpath[1] - wpath[0]
            target_ori /= torch.linalg.norm(target_ori)
            v = torch.cross(b_ori, target_ori)
            c = torch.dot(b_ori, target_ori)
            s = torch.linalg.norm(v)
            kmat = torch.cuda.FloatTensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            target_rot = torch.eye(3).cuda() + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
            target_rot = target_rot.unsqueeze(0).repeat(2, 1, 1)
            pelvis_zero = self.bm_2frame(betas=motion_seed_dict['betas']).joints[:1, 0, :]  # [1, 3]
            original_rot = pytorch3d.transforms.axis_angle_to_matrix(motion_seed_dict['global_orient'])
            new_rot = torch.einsum('bij,bjk->bik', target_rot, original_rot)
            new_transl = torch.einsum('bij,bj->bi', target_rot, pelvis_zero + motion_seed_dict['transl']) - pelvis_zero
            motion_seed_dict['global_orient'] = pytorch3d.transforms.matrix_to_axis_angle(new_rot)
            motion_seed_dict['transl'] = new_transl

            # translate to make the init body pelvis above origin, feet on floor
            output = self.bm_2frame(**motion_seed_dict)
            transl = torch.cuda.FloatTensor([output.joints[0, 0, 0], output.joints[0, 0, 1], output.joints[0, :, 2].amin()])
            motion_seed_dict['transl'] -= transl
            motion_seed_dict['transl'] += wpath[:1]

        output = self.bm_2frame(**motion_seed_dict)
        wpath[0] = output.joints[0,0,:]
        wpath[1,2] = wpath[0,2]

        """generate a body"""
        xbo_dict = {}
        xbo_dict['motion_seed'] = motion_seed_dict
        xbo_dict['betas'] = motion_seed_dict['betas'][:1, :]
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        # xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
        #                                output_type='aa').view(1, -1)
        # xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]

        """snap to the ground"""
        # bm = self.bm_male if gender == 'male' else self.bm_female
        # bm = self.bm_male
        # xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        # xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        # wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        # wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = self.navmesh_path
        xbo_dict['floor_height'] = 0
        # a precomputed path len using path finding to calc the max depth for rl
        # xbo_dict['path_len'] = path_len
        if shapely_poly is not None:
            xbo_dict['shapely_poly'] = shapely_poly
        self.index_rec += 1


        if visualize:
            init_body1_mesh = trimesh.Trimesh(
                vertices=self.bm_2frame(**motion_seed_dict).vertices[0].detach().cpu().numpy(),
                faces=self.bm_2frame.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            init_body2_mesh = trimesh.Trimesh(
                vertices=self.bm_2frame(**motion_seed_dict).vertices[1].detach().cpu().numpy(),
                faces=self.bm_2frame.faces,
                vertex_colors=np.array([100, 100, 150])
            )
            # floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
            #                                   transform=np.array([[1.0, 0.0, 0.0, 0],
            #                                                       [0.0, 1.0, 0.0, 0],
            #                                                       [0.0, 0.0, 1.0, -0.005],
            #                                                       [0.0, 0.0, 0.0, 1.0],
            #                                                       ]),
            #                                   )
            # floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(mesh_path, force='mesh')
            obj_mesh.vertices[:, 2] -= 0.02
            vis_mesh = [
                # floor_mesh,
                        init_body1_mesh,
                        init_body2_mesh,
                        obj_mesh, navmesh,
                        trimesh.creation.axis()
                        ]

            for point_idx, pelvis in enumerate(wpath[:2, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        # pdb.set_trace()
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict


class Egobody(BatchGeneratorReachingTarget):
    def __init__(self, scene_dir=None, body_model_path="/home/yzhang/body_models/VPoser"):
        super().__init__("", body_model_path, "ssm2_67")
        self.scene_dir = Path(scene_dir)
        self.navmesh_loose = trimesh.load(os.path.join(self.scene_dir, "navmesh_tight.ply"), force='mesh')
        self.navmesh_looser = trimesh.load(os.path.join(self.scene_dir, "navmesh_looser.ply"), force='mesh')
        walkable_region = []
        verts = self.navmesh_loose.vertices[:,:2]
        for face in self.navmesh_loose.faces:
            walkable_region.append(Polygon([verts[face[0]], verts[face[1]], verts[face[2]]]))
        self.walkable_region = union_all(walkable_region)
        # only walk on the biggest area 
        if type(self.walkable_region)==shapely.geometry.multipolygon.MultiPolygon:
            area = [x.area for x in self.walkable_region.geoms]
            idx = area.index(max(area))
            self.walkable_region = self.walkable_region.geoms[idx]
    
    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                    scene_idx=None, start_target=None, random_rotation_range=1.0,
                    clip_far=False,res=32, extent=1.6):
        while True:
            start = np.array(trimesh.sample.sample_surface_even(self.navmesh_looser, 1)[0]) 
            target = np.array(trimesh.sample.sample_surface_even(self.navmesh_looser, 1)[0])
            if self.walkable_region.contains(Point(start[0][:2])) and \
               np.linalg.norm(target - start) >= 2 and \
               np.linalg.norm(target - start) <= 5 and \
               self.walkable_region.contains(Point(target[0][:2])):
                break
        wpath = torch.cuda.FloatTensor(np.zeros((3,3)))
        wpath[0] = torch.cuda.FloatTensor(start)
        wpath[1] = torch.cuda.FloatTensor(target)
        wpath[2] = torch.cuda.FloatTensor(target)
        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = 'male'
        xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]

        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['floor_height'] = 0
        self.index_rec += 1
        xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['navmesh'] = self.navmesh_loose
        xbo_dict['scene_path'] = os.path.join(self.scene_dir, 'mesh_floor_zup.ply')
        xbo_dict['navmesh_path'] = os.path.join(self.scene_dir, 'navmesh_tight.ply')
        xbo_dict['valid_region'] = self.walkable_region

        """generate a body"""
        """
        xbo_dict2 = {}
        wpath2 = torch.cuda.FloatTensor(np.zeros((3,3)))
        wpath2[0] = torch.cuda.FloatTensor(target)
        wpath2[1] = torch.cuda.FloatTensor(start)
        wpath2[2] = torch.cuda.FloatTensor(start)
        xbo_dict2['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        # xbo_dict2['betas'] = torch.cuda.FloatTensor(1,10).zero_()
        xbo_dict2['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict2['global_orient'] = self.get_bodyori_from_wpath(wpath2[0], wpath2[-1])[None,...]

        '''snap to the ground'''
        xbo_dict2['transl'] = wpath2[:1] - bm(**xbo_dict2).joints[0, 0, :]  # [1,3]
        xbo_dict2 = self.snap_to_ground(xbo_dict2, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath2[0] = bm(**xbo_dict2).joints[0, 0, :]
        wpath2[1, 2] = wpath2[0, 2]

        '''specify output'''
        xbo_dict2['gender']=gender
        xbo_dict2['wpath']=wpath2[:2]
        if torch.isnan(xbo_dict2['wpath']).any() or torch.isinf(xbo_dict2['wpath']).any():
            print('error:wpath invalid', xbo_dict2['wpath'])
        xbo_dict2['floor_height'] = 0
        self.index_rec += 1
        xbo_dict2['betas'] = xbo_dict2['betas'][0]
        xbo_dict2['navmesh'] = self.navmesh_loose
        xbo_dict2['scene_path'] = os.path.join(self.scene_dir, 'mesh_floor_zup.ply')
        xbo_dict2['navmesh_path'] = os.path.join(self.scene_dir, 'navmesh_looser.ply')
        xbo_dict2['valid_region'] = self.walkable_region
        return xbo_dict, xbo_dict2
        """
        return xbo_dict


class Egobody(BatchGeneratorScene2frameTrain):
    def __init__(self, dataset_path, body_model_path='/home/yzhang/body_models/VPoser', body_repr='ssm2_67', motion_seed_list=None, scene_list=None, scene_dir=None, scene_type='random'):
        super().__init__(dataset_path, body_model_path, body_repr, motion_seed_list, scene_list, scene_dir, scene_type)
        self.navmesh = trimesh.load(os.path.join(self.scene_dir, "navmesh_tight.ply"), force='mesh')
        walkable_region = []
        verts = self.navmesh.vertices[:,:2]
        for face in self.navmesh.faces:
            walkable_region.append(Polygon([verts[face[0]], verts[face[1]], verts[face[2]]]))
        self.walkable_region = union_all(walkable_region)
        # only walk on the biggest area
        if type(self.walkable_region) == shapely.geometry.multipolygon.MultiPolygon:
            area = [x.area for x in self.walkable_region.geoms]
            idx = area.index(max(area))
            self.walkable_region = self.walkable_region.geoms[idx]

        self.bm_male = smplx.create(body_model_path, model_type='smplx',
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
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
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

        self.motion_data = np.load("data/locomotion/subseq_00343.npz")
        
    def gen_init_body(self, start, target, gender):
        wpath = np.zeros((3,3))
        wpath = torch.cuda.FloatTensor(wpath)
        wpath[0] = torch.cuda.FloatTensor(start)
        wpath[1] = torch.cuda.FloatTensor(target)
        wpath[2] = torch.cuda.FloatTensor(target)

        # start_frame = 5
        motion_data = self.motion_data
        start_frame = torch.randint(0, len(motion_data['poses']) - 1, (1,)).item()
        motion_seed_dict = {}
        # motion_seed_dict['betas'] = torch.cuda.FloatTensor(motion_data['betas']).reshape((1, 10)).repeat(2, 1)
        motion_seed_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_(std=0.1).repeat(2, 1)
        motion_seed_dict['body_pose'] = torch.cuda.FloatTensor(motion_data['poses'][start_frame:start_frame + 2, 3:66])
        motion_seed_dict['global_orient'] = torch.cuda.FloatTensor(motion_data['poses'][start_frame:start_frame + 2, :3])
        motion_seed_dict['transl'] = torch.cuda.FloatTensor(motion_data['trans'][start_frame:start_frame + 2])

        # rotate the body to face the target
        if gender == 'male':
            bm = self.bm_male
        elif gender == "female":
            bm = self.bm_female
        else:
            bm = None
            pdb.set_trace()

        # rotate the body to face the target
        joints = bm(**motion_seed_dict).joints
        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clip(min=1e-12)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=x_axis.device).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        b_ori = y_axis[0]
        b_ori /= torch.linalg.norm(b_ori)
        target_ori = wpath[1] - wpath[0]
        target_ori /= torch.linalg.norm(target_ori)
        v = torch.cross(b_ori, target_ori)
        c = torch.dot(b_ori, target_ori)
        s = torch.linalg.norm(v)
        kmat = torch.cuda.FloatTensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        target_rot = torch.eye(3).cuda() + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
        target_rot = target_rot.unsqueeze(0).repeat(2, 1, 1)
        pelvis_zero = bm(betas=motion_seed_dict['betas']).joints[:1, 0, :]  # [1, 3]
        original_rot = pytorch3d.transforms.axis_angle_to_matrix(motion_seed_dict['global_orient'])
        new_rot = torch.einsum('bij,bjk->bik', target_rot, original_rot)
        new_transl = torch.einsum('bij,bj->bi', target_rot, pelvis_zero + motion_seed_dict['transl']) - pelvis_zero
        motion_seed_dict['global_orient'] = pytorch3d.transforms.matrix_to_axis_angle(new_rot)
        motion_seed_dict['transl'] = new_transl
        # slightly rotate around z
        theta = torch.cuda.FloatTensor(1).uniform_(-1, 1) * torch.pi * 2 * 0.1
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
                                                                 convention="XYZ").reshape(1, 3, 3)
        original_rot = pytorch3d.transforms.axis_angle_to_matrix(motion_seed_dict['global_orient'])
        new_rot = torch.einsum('bij,bjk->bik', random_rot, original_rot)
        new_transl = torch.einsum('bij,bj->bi', random_rot, pelvis_zero + motion_seed_dict['transl']) - pelvis_zero
        motion_seed_dict['global_orient'] = pytorch3d.transforms.matrix_to_axis_angle(new_rot)
        motion_seed_dict['transl'] = new_transl

        # translate to make the init body pelvis above origin, feet on floor
        output = bm(**motion_seed_dict)
        transl = torch.cuda.FloatTensor([output.joints[0, 0, 0], output.joints[0, 0, 1], output.joints[0, :, 2].amin()])
        motion_seed_dict['transl'] -= transl
        motion_seed_dict['transl'] += wpath[:1]
        output = bm(**motion_seed_dict)
        wpath[0] = output.joints[0,0,:]
        wpath[1,2] = wpath[0,2]

        """generate a body"""
        xbo_dict = {}
        xbo_dict['motion_seed'] = motion_seed_dict
        xbo_dict['betas'] = motion_seed_dict['betas'][:1, :]
        xbo_dict['gender'] = gender
        xbo_dict['wpath'] = wpath[:2]
        xbo_dict['scene_path'] = os.path.join(self.scene_dir, "mesh_floor_zup.ply")
        xbo_dict['navmesh'] = self.navmesh
        xbo_dict['navmesh_path'] = os.path.join(self.scene_dir, "navmesh_tight.ply")
        xbo_dict['floor_height'] = 0
        xbo_dict['shapely_poly'] = self.walkable_region
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict
    
    def next_body(self):
        while True:
            while True:
                start = trimesh.sample.sample_surface_even(self.navmesh, 1)[0]
                if self.walkable_region.contains(Point(start[0][:2]).buffer(0.3)):
                    break
            while True:
                target = trimesh.sample.sample_surface_even(self.navmesh, 1)[0]
                if self.walkable_region.contains(Point(target[0][:2]).buffer(0.3)):
                    break
            if np.linalg.norm(target - start) >= 2 and np.linalg.norm(target - start) <= 5:
                break
        gender = random.choice(['male', 'female'])
        xbo_dict = self.gen_init_body(start, target, gender)
        return xbo_dict


class BatchGeneratorSceneTrain(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 scene_list=None,
                 scene_dir=None,
                 scene_type='random',
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene_list= scene_list
        self.scene_dir = Path(scene_dir)
        self.scene_idx = 0
        self.scene_type=scene_type
        # with open(os.path.join(dataset_path, 'orient.json')) as f:
        #     self.orient = np.array(json.load(f)).reshape(1, 3)

    # TODO: change to random wpath gen!
    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                  scene_idx=None, start_target=None, random_rotation_range=1.0,
                  clip_far=False,
                  res=32, extent=1.6):
        if scene_idx is None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        scene_name = self.scene_list[scene_idx]
        if self.scene_type == 'prox':
            mesh_path = self.scene_dir / 'PROX' / (scene_name + '_floor.ply')
            navmesh_path = self.scene_dir / 'PROX' / (scene_name + '_navmesh.ply')
        elif self.scene_type == 'room_0':
            mesh_path = self.scene_dir / 'mesh_floor.ply'
            navmesh_path = self.scene_dir / 'navmesh_tight.ply'
            samples_path = self.scene_dir / 'samples.pkl'
        elif 'random' in self.scene_type:
            mesh_path = self.scene_dir / self.scene_type / (scene_name + '.ply')
            navmesh_path = self.scene_dir / self.scene_type / (scene_name + '_navmesh_tight.ply')
            samples_path = self.scene_dir / self.scene_type / (scene_name + '_samples.pkl')
            # navmesh_loose_path = self.scene_dir / self.scene_type / (scene_name + '_navmesh.ply')
            # navmesh_loose = trimesh.load(navmesh_loose_path, force='mesh')
        navmesh = trimesh.load(navmesh_path, force='mesh')
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 50])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )
        wpath = np.zeros((3,3))
        wpath = torch.cuda.FloatTensor(wpath)

        # scene = pyrender.Scene()
        # scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
        # # scene.add(pyrender.Mesh.from_trimesh(navmesh, smooth=False))
        # scene.add(pyrender.Mesh.from_trimesh(navmesh_crop, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        """randomly specify a 3D path"""
        wpath = np.zeros((3,3))
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        if start_target is not None:
            wpath[0] = torch.cuda.FloatTensor(start_target[0])  # starting point
            wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
        elif self.scene_type == 'prox':
            start_target = np.zeros((2, 3))  # pairs of start and target positions
            max_try = 32
            for try_idx in range(max_try):
                start_target[0] = trimesh.sample.sample_surface_even(navmesh, 1)[0]
                if try_idx < max_try - 1:
                    crop_box = trimesh.creation.box(extents=[sigma, sigma, 2])
                    crop_box.vertices += start_target[0]
                    navmesh_crop = navmesh.slice_plane(crop_box.facets_origin, -crop_box.facets_normal)
                    if len(navmesh_crop.vertices) >= 3:
                        start_target[1] = trimesh.sample.sample_surface_even(navmesh_crop, 1)[0]
                        break
                else:
                    start_target[1] = trimesh.sample.sample_surface_even(navmesh, 1)[0]

            if np.isnan(start_target).any():
                print('error in sampling start-target')
            wpath[0] = torch.cuda.FloatTensor(start_target[0]) #starting point
            wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
            if torch.isnan(wpath).any() or torch.isinf(wpath).any():
                print('error:wpath invalid, random sample', wpath)
        elif 'random' in self.scene_type or self.scene_type == 'room_0':
            with open(samples_path, 'rb') as f:
                sample_pairs = pickle.load(f)
            num_samples = len(sample_pairs)
            if num_samples == 0:
                print('error: zero samples, precompute')
            start, target = sample_pairs[np.random.randint(low=0, high=num_samples)]
            # random wpath sampling 
            # start = np.array(trimesh.sample.sample_surface_even(navmesh_loose, 1)[0])
            # target = np.array(trimesh.sample.sample_surface_even(navmesh_loose, 1)[0])

            if clip_far and np.linalg.norm(target - start) > 1.0:
                # print('clip far pairs')
                length = np.linalg.norm(target - start).clip(min=1e-12)
                vec_dir = (target - start) / length
                l1 = np.random.uniform(low=0.0, high=length - 0.5)
                l2 = min(np.random.uniform(0.5, 1.0) + l1, length)
                target = start + vec_dir * l2
                start = start + vec_dir * l1
            wpath[0] = torch.cuda.FloatTensor(start)  # starting point
            wpath[1] = torch.cuda.FloatTensor(target)  # ending point xy
            if torch.isnan(wpath).any() or torch.isinf(wpath).any():
                print('error:wpath invalid, precompute', wpath)

        # wpath[2, :2] = wpath[0, :2] + torch.randn(2).to(device=wpath.device) #point to initialize the body orientation, not returned
        theta = torch.pi * (2 * torch.cuda.FloatTensor(1).uniform_() - 1) * random_rotation_range
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                                  convention="XYZ")
        wpath[2] = torch.einsum('ij, j->i', random_rotz[0], wpath[1] - wpath[0]) + wpath[0]  # face the target with [-90, 90] disturbance
        if torch.norm(wpath[2] - wpath[0], dim=-1) < 1e-12:
            wpath[2] += 1e-12
        # hard code
        # wpath[0] = torch.cuda.FloatTensor([-1, 0, 0])
        # wpath[1] = torch.cuda.FloatTensor([0.5, 0, 0])
        # wpath[2] = wpath[1]

        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]

        """snap to the ground"""
        # bm = self.bm_male if gender == 'male' else self.bm_female
        bm = self.bm_male
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        xbo_dict['floor_height'] = 0
        self.index_rec += 1

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            # floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
            #                                   transform=np.array([[1.0, 0.0, 0.0, 0],
            #                                                       [0.0, 1.0, 0.0, 0],
            #                                                       [0.0, 0.0, 1.0, -0.005],
            #                                                       [0.0, 0.0, 0.0, 1.0],
            #                                                       ]),
            #                                   )
            # floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(mesh_path, force='mesh')
            obj_mesh.vertices[:, 2] -= 0.02
            vis_mesh = [
                # floor_mesh,
                        init_body_mesh,
                        obj_mesh, navmesh,
                        trimesh.creation.axis()
                        ]

            for point_idx, pelvis in enumerate(wpath[:2, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            _, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 100])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

class CrowdMotion(BatchGeneratorSceneTrain):
    def __init__(self, dataset_path, body_model_path='/home/yzhang/body_models/VPoser', body_repr='ssm2_67', scene_list=None, scene_dir=None, scene_type='random',):
        super().__init__(dataset_path, body_model_path, body_repr, scene_list, scene_dir, scene_type)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                    scene_idx=None, start_target=None, random_rotation_range=1.0,
                    clip_far=False,
                    res=32, extent=1.6):
        if scene_idx is None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        scene_name = self.scene_list[scene_idx]
        mesh_path = self.scene_dir / self.scene_type / (scene_name + '.ply')
        navmesh_path = self.scene_dir / self.scene_type / (scene_name + '_navmesh_tight.ply')
        samples_path = self.scene_dir / self.scene_type / (scene_name + '_samples.pkl')
        navmesh = trimesh.load(navmesh_path, force='mesh')
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 50])
        with open(samples_path, 'rb') as f:
            sample_pairs = pickle.load(f)
            num_samples = len(sample_pairs)
            if num_samples == 0:
                print('error: zero samples, precompute')
            start, target = sample_pairs[np.random.randint(low=0, high=num_samples)]

        wpath = torch.cuda.FloatTensor(np.zeros((3,3)))
        wpath[0] = torch.cuda.FloatTensor(start)
        wpath[1] = torch.cuda.FloatTensor(target)
        wpath[2] = torch.cuda.FloatTensor(target)
        """generate a body"""
        xbo_dict = {}
        gender = 'male'
        xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]

        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['floor_height'] = 0
        self.index_rec += 1
        xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['navmesh'] = navmesh
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['navmesh_path'] = navmesh_path
        
        xbo_dict2 = {}
        wpath2 = torch.cuda.FloatTensor(np.zeros((3,3)))
        wpath2[0] = torch.cuda.FloatTensor(target)
        wpath2[1] = torch.cuda.FloatTensor(start)
        wpath2[2] = torch.cuda.FloatTensor(start)
        xbo_dict2['betas'] = torch.cuda.FloatTensor(1,10).zero_()
        xbo_dict2['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict2['global_orient'] = self.get_bodyori_from_wpath(wpath2[0], wpath2[-1])[None,...]

        '''snap to the ground'''
        xbo_dict2['transl'] = wpath2[:1] - bm(**xbo_dict2).joints[0, 0, :]  # [1,3]
        xbo_dict2 = self.snap_to_ground(xbo_dict2, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath2[0] = bm(**xbo_dict2).joints[0, 0, :]
        wpath2[1, 2] = wpath2[0, 2]

        '''specify output'''
        xbo_dict2['gender']=gender
        xbo_dict2['wpath']=wpath2[:2]
        if torch.isnan(xbo_dict2['wpath']).any() or torch.isinf(xbo_dict2['wpath']).any():
            print('error:wpath invalid', xbo_dict2['wpath'])
        xbo_dict2['floor_height'] = 0
        self.index_rec += 1
        xbo_dict2['betas'] = xbo_dict2['betas'][0]
        xbo_dict2['navmesh'] = navmesh
        xbo_dict2['scene_path'] = mesh_path
        xbo_dict2['navmesh_path'] = navmesh_path

        return xbo_dict, xbo_dict2

class BatchGeneratorSceneRandomTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 scene_list=None,
                 scene_dir=None,
                 scene_type='random',
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene_list= scene_list
        self.scene_dir = Path(scene_dir)
        self.scene_idx = 0
        self.scene_type=scene_type
        # with open(os.path.join(dataset_path, 'orient.json')) as f:
        #     self.orient = np.array(json.load(f)).reshape(1, 3)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                  scene_idx=None, wpath=None, path_idx=None,
                  clip_far=False,
                  res=32, extent=1.6):
        if scene_idx is None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        scene_name = self.scene_list[scene_idx]
        if self.scene_type == 'prox':
            mesh_path = self.scene_dir / 'PROX' / (scene_name + '_floor.ply')
            navmesh_path = self.scene_dir / 'PROX' / (scene_name + '_navmesh.ply')
        elif self.scene_type == 'random':
            mesh_path = self.scene_dir / 'random_scene' / (scene_name + '.ply')
            navmesh_path = self.scene_dir / 'random_scene' / (scene_name + '_navmesh_tight.ply')
            samples_path = self.scene_dir / 'random_scene' / (scene_name + '_samples.pkl')
        elif self.scene_type == 'random_obstacle':
            mesh_path = self.scene_dir / 'random_scene_obstacle' / (scene_name + '.ply')
            navmesh_path = self.scene_dir / 'random_scene_obstacle' / (scene_name + '_navmesh_tight.ply')
            samples_path = self.scene_dir / 'random_scene_obstacle' / (scene_name + '_samples.pkl')
        navmesh = trimesh.load(navmesh_path, force='mesh')
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 200])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )

        # import pyrender
        # scene = pyrender.Scene()
        # scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
        # # scene.add(pyrender.Mesh.from_trimesh(navmesh, smooth=False))
        # scene.add(pyrender.Mesh.from_trimesh(navmesh_crop, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        """randomly specify a 3D path"""
        if wpath is not None:
            wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        elif self.scene_type in ['random']:
            with open(samples_path, 'rb') as f:
                sample_pairs = pickle.load(f)
            print(sample_pairs)
            paths = []
            last_point = None
            path = []
            for sample in sample_pairs:
                if last_point is not None and not np.array_equal(last_point, sample[0]):
                    paths.append(path)
                    path = [sample[0]]
                elif last_point is None:
                    path = [sample[0]]
                last_point = sample[1]
                path.append(last_point)
            if last_point is not None:
                paths.append(path)
            print('#path:', len(paths))
            path_idx = random.choice(range(len(paths))) if path_idx is None else path_idx
            path_name = 'path' + str(path_idx)
            print(paths[path_idx])
            wpath = np.stack(paths[path_idx], axis=0)
            wpath = torch.cuda.FloatTensor(wpath)
            print(wpath.shape)

        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None,...]

        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1:, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        xbo_dict['path_name'] = path_name
        xbo_dict['floor_height'] = 0
        self.index_rec += 1

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            # floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
            #                                   transform=np.array([[1.0, 0.0, 0.0, 0],
            #                                                       [0.0, 1.0, 0.0, 0],
            #                                                       [0.0, 0.0, 1.0, -0.005],
            #                                                       [0.0, 0.0, 0.0, 1.0],
            #                                                       ]),
            #                                   )
            # floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(mesh_path, force='mesh')
            obj_mesh.vertices[:, 2] -= 0.2
            vis_mesh = [
                # floor_mesh,
                        init_body_mesh,
                        obj_mesh, navmesh,
                        trimesh.creation.axis()
                        ]

            for point_idx, pelvis in enumerate(wpath[:, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            _, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 100])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

class BatchGeneratorSceneTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)

    def interpolate_path(self, wpath):
        interpolated_path = [wpath[0]]
        last_point = wpath[0]
        for point_idx in range(1, wpath.shape[0]):
            while torch.norm(wpath[point_idx] - last_point) > 1:
                last_point = last_point + (wpath[point_idx] - last_point) / torch.norm(
                    wpath[point_idx] - last_point)
                interpolated_path.append(last_point)
            last_point = wpath[point_idx]
            interpolated_path.append(last_point)
        return torch.stack(interpolated_path, dim=0)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True, use_zero_shape=True,
                  scene_path=None, floor_height=0, navmesh_path=None,
                  wpath_path=None, path_name=None,
                  last_motion_path=None,
                  clip_far=False, random_orient=False,
                  res=32, extent=1.6):

        """get navmesh"""
        if navmesh_path.exists():
            navmesh = trimesh.load(navmesh_path, force='mesh')
        else:
            from test_navmesh import create_navmesh, zup_to_shapenet
            scene_mesh = trimesh.load(scene_path, force='mesh')
            """assume the scene coords are z-up"""
            scene_mesh.vertices[:, 2] -= floor_height
            scene_mesh.apply_transform(zup_to_shapenet)
            navmesh = create_navmesh(scene_mesh, export_path=navmesh_path, agent_radius=0.01, visualize=False)
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 50])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )

        """get wpath"""
        with open(wpath_path, 'rb') as f:
            wpath = pickle.load(f)  # [n, 3]
        wpath = torch.cuda.FloatTensor(wpath)
        if clip_far:
            wpath = self.interpolate_path(wpath)

        """load or generate a body"""
        xbo_dict = {}
        if last_motion_path is not None:
            with open(last_motion_path, 'rb') as f:
                motion_data = pickle.load(f)  # [n, 3]
            last_primitive = motion_data['motion'][-1]
            gender = last_primitive['gender']
            betas = xbo_dict['betas'] = torch.cuda.FloatTensor(last_primitive['betas']).reshape((1, 10))
            smplx_params = torch.cuda.FloatTensor(last_primitive['smplx_params'][0, -1:])
            R0 = torch.cuda.FloatTensor(last_primitive['transf_rotmat'])
            T0 = torch.cuda.FloatTensor(last_primitive['transf_transl'])
            from models.baseops import SMPLXParser
            pconfig_1frame = {
                'n_batch': 1,
                'device': 'cuda',
                'marker_placement': 'ssm2_67'
            }
            smplxparser_1frame = SMPLXParser(pconfig_1frame)
            smplx_params = smplxparser_1frame.update_transl_glorot(R0.permute(0, 2, 1),
                                                                   -torch.einsum('bij,bkj->bki', R0.permute(0, 2, 1),
                                                                                 T0),
                                                                   betas=betas,
                                                                   gender=gender,
                                                                   xb=smplx_params,
                                                                   inplace=False,
                                                                   to_numpy=False)  # T0 must be [1, 1, 3], [1,3] leads to error
            xbo_dict['transl'] = smplx_params[:, :3]
            xbo_dict['global_orient'] = smplx_params[:, 3:6]
            xbo_dict['body_pose'] = smplx_params[:, 6:69]
            bm = self.bm_male if gender == 'male' else self.bm_female
        else:
            # gender = random.choice(['male', 'female'])
            gender = random.choice(['male'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_() if use_zero_shape else torch.cuda.FloatTensor(1,10).normal_()
            """maunal rest pose"""
            # body_pose = torch.zeros(1, 63).to(dtype=torch.float32)
            # body_pose[:, 45:48] = -torch.tensor([0, 0, 1]) * torch.pi * 0.45
            # body_pose[:, 48:51] = torch.tensor([0, 0, 1]) * torch.pi * 0.45
            # xbo_dict['body_pose'] = body_pose.to(device='cuda')
            xbo_dict['body_pose'] = torch.cuda.FloatTensor(rest_pose) if use_zero_pose else self.vposer.decode(torch.cuda.FloatTensor(1,32).normal_(), output_type='aa').view(1, -1)
            if random_orient:
                target = wpath[0] + torch.cuda.FloatTensor(3).normal_()
                target[2] = wpath[0, 2]
                # xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], target)[None,...]
                xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[0] + torch.cuda.FloatTensor([-1, 0, 0]))[None, ...]
            else:
                xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None, ...]
            """snap to the ground"""
            bm = self.bm_male if gender == 'male' else self.bm_female
            xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
            xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        init_body = bm(**xbo_dict)
        wpath[0] = init_body.joints[0, 0]
        start_markers = init_body.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]
        wpath[1:, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = scene_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        xbo_dict['path_name'] = path_name
        xbo_dict['floor_height'] = floor_height
        xbo_dict['motion_history'] = motion_data if last_motion_path is not None else None

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.0051],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(scene_path, force='mesh')
            obj_mesh.vertices[:, 2] -= floor_height + 0.05
            vis_mesh = [
                floor_mesh,
                        init_body_mesh,
                        obj_mesh,
                navmesh,
                        trimesh.creation.axis()
                        ]

            marker_meshes = []
            for point_idx, pelvis in enumerate(wpath[:, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat, axis_radius=0, origin_size=0.05, origin_color=np.array([0, 200, 0]))
                vis_mesh.append(point_axis)
                if point_idx > 0:
                    marker_meshes.append(point_axis)


            marker_dirs = wpath[1] - start_markers
            marker_dirs = marker_dirs / torch.norm(marker_dirs, keepdim=True, dim=-1)
            for marker_idx in range(start_markers.reshape(-1, 3).shape[0]):
                marker = start_markers.reshape(-1, 3)[marker_idx]
                marker_dir = marker_dirs.reshape(-1, 3)[marker_idx]
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
                marker_meshes.append(sm)
                grad_vec = trimesh.creation.cylinder(radius=0.002, segment=np.stack(
                    [marker.detach().cpu().numpy(),
                     (marker + 0.1 * marker_dir).detach().cpu().numpy()]))
                grad_vec.visual.vertex_colors = np.array([0, 0, 255, 255])
                marker_meshes.append(grad_vec)
            # trimesh.util.concatenate(marker_meshes).show()

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            _, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 200])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1.5), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict
