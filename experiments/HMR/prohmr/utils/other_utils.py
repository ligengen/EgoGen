import torch

def coord_transf(transf_mtx, points):
    '''
    transform coordinates from a coordinate (coord1) system to another (coord2)

    args:
        transf_mtx: [batch, 4, 4]: 4x4 SE3 transformation matrices from coord1 to coord2
        points: [batch, N_points, 3]: xyz coordinates of N points in coord system 1
    
    outputs:
        points_transformed: xyz coordinates in coord system 2
    '''
    batch_size = len(points)

    transf_mtx = transf_mtx.unsqueeze(1) # [bs, 1, 4, 4]

    homogen_coord = torch.ones([batch_size, points.shape[1], 1], dtype=points.dtype, device=points.device)
    points_homo = torch.cat([points, homogen_coord], dim=2)

    points_t_homo = torch.matmul(transf_mtx, points_homo.unsqueeze(-1))
    points_transformed = points_t_homo[:,:,:3,0]

    return points_transformed

def coord_multiple_transf(transf_mtx1, transf_mtx2, points):
    '''
    when transforming points through 2 transformation matrices, it's more efficient
    to compute the total transformation matrix first and then apply to the points

    NOTE the order of applying the transformation matrices: T2*T1*points will apply T1 first
    '''
    total_transf = torch.matmul(transf_mtx2, transf_mtx1)
    
    points_transformed = coord_transf(total_transf, points)
    return points_transformed

def coord_transf_holo_yz_reverse(points):
    '''
    inside hololense the y and z axis are negative to what's defined by kinect
    this function simply reverse it
    '''

    batch_size = points.shape[0]
    T = torch.diag(torch.tensor([1, -1, -1, 1])).repeat(batch_size, 1, 1)
    T = T.to(points.device).type(points.dtype)

    points_transformed = coord_transf(T, points)
    return points_transformed

