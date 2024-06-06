import torch
import numpy as np
from utils import lbs as my_lbs


class GarmentSMPL:
    """
    Generates garment geometry in the given poses using SMPL blend shapes and linear blend skinning.
    """

    def __init__(self, smpl_model, garment_skinning_dict):
        self.smpl_model = smpl_model
        self.garment_skinning_dict = garment_skinning_dict
        self.A_pose = torch.FloatTensor(np.zeros((1, 165)))
        self.A_pose[:, 50] = -0.7853981633974483
        self.A_pose[:, 53] = 0.7853981633974483

    def make_vertices(self, betas: torch.FloatTensor, full_pose: torch.FloatTensor,
                      transl: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Generates garment vertices for the given pose and shape parameters.
        :param betas: [Nx10] shape parameters
        :param full_pose: [Nx72] pose parameters in axis-angle format
        :param transl: [Nx3] grobal translation
        :return: [NxVx3] garment vertices
        """
        J_transformed, joint_transforms = my_lbs.get_transformed_joints(betas, full_pose, self.smpl_model.v_template,
                                                                        self.smpl_model.shapedirs,
                                                                        self.smpl_model.J_regressor,
                                                                        self.smpl_model.parents)
        if self.A_pose.shape[0] != full_pose.shape[0]:
            self.A_pose = self.A_pose.repeat(full_pose.shape[0], 1)
        A_J_transformed, A_joint_transforms = my_lbs.get_transformed_joints(betas, self.A_pose, self.smpl_model.v_template,
                                                                        self.smpl_model.shapedirs,
                                                                        self.smpl_model.J_regressor,
                                                                        self.smpl_model.parents)
        v_garment, _ = my_lbs.pose_garment(betas, full_pose, self.garment_skinning_dict['v'],
                                           self.garment_skinning_dict['shapedirs'],
                                           self.garment_skinning_dict['posedirs'],
                                           self.garment_skinning_dict['lbs_weights'], J_transformed, joint_transforms, A_joint_transforms=A_joint_transforms, A_POSE=self.A_pose)

        if transl is not None:
            v_garment = v_garment + transl[:, None]

        return v_garment
