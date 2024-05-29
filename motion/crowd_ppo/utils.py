import os
import pickle
import datetime
import calendar
import time
import torch
import pdb
import torch.nn.functional as F

def save_rollout_results(scene, outmps, outfolder, man_id=None):
    # print(outfolder)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    mp_keys = ['blended_marker', 'smplx_params', 'betas', 'gender', 'transf_rotmat', 'transf_transl', 'pelvis_loc',
               'mp_type']

    wpath = scene['wpath']
    for b in range(1):
        outmps_nodes = {'motion': [], 'wpath': wpath.detach().cpu().numpy(),
                        # 'target_orient': scene['target_orient'].detach().cpu().numpy(),
                        'navmesh_path': scene['navmesh_path'],

                        }
        if 'obj_id' in scene:
            outmps_nodes['obj_id'] = scene['obj_id']
        if 'obj_transform' in scene:
            outmps_nodes['obj_transform'] = scene['obj_transform'],
        if 'scene_path' in scene:
            outmps_nodes['scene_path'] = scene['scene_path']
        for mp in outmps:
            mp_node = {}
            for idx, key in enumerate(mp_keys):
                if key in ['gender', 'mp_type', 'betas', 'transf_rotmat', 'transf_transl']:
                    mp_node[key] = mp[idx] if type(mp[idx]) == str else mp[idx].detach().cpu().numpy()
                elif key in ['smplx_params']:
                    mp_node[key] = mp[idx][b:b + 1].detach().cpu().numpy()
                else:
                    mp_node[key] = mp[idx][b].detach().cpu().numpy()
            outmps_nodes['motion'].append(mp_node)
        current_GMT = time.gmtime()
        ts = calendar.timegm(current_GMT)
        # with open(outfolder + '/motion_%d.pkl' % (ts % 500000), 'wb') as f:
        if man_id is None:
            name = '/motion_%s.pkl' % str(time.time())
            with open(outfolder + name, 'wb') as f:
                pickle.dump(outmps_nodes, f)
            f.close()
        else:
            with open(outfolder + '/motion_%s.pkl' % man_id, 'wb') as f:
                pickle.dump(outmps_nodes, f)
            f.close()


def calc_sdf(vertices, sdf_dict, return_gradient=False):
    sdf_centroid = sdf_dict['center']
    sdf_scale = sdf_dict['scale']
    sdf_grids = sdf_dict['sdf']
    sdf_grids = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0)
    sdf_centroid = sdf_centroid.reshape(1, 1, 3)

    batch_size, num_vertices, _ = vertices.shape
    vertices = vertices.reshape(1, -1, 3)  # [B, V, 3]
    vertices = (vertices - sdf_centroid) * sdf_scale  # convert to [-1, 1]
    sdf_values = F.grid_sample(sdf_grids,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   # align_corners=True
                                   ).reshape(batch_size, num_vertices)
    """
    if return_gradient:
        sdf_gradients = sdf_dict['gradient_grid']
        gradient_values = F.grid_sample(sdf_gradients,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3),
                                   # [2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   ).permute(2, 1, 0, 3, 4).reshape(batch_size, num_vertices, 3)
        gradient_values = gradient_values / torch.norm(gradient_values, dim=-1, keepdim=True).clip(min=1e-12)
        return sdf_values, gradient_values
    """

    # indoor scenes < 0 no pene
    return -sdf_values

