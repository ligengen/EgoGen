import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from yacs.config import CfgNode

class TranslEnc(nn.Module):

    def __init__(self, in_dim=3, out_dim=128):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(TranslEnc, self).__init__()
        # self.cfg = cfg
        # self.npose = 6 * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.layers = nn.Sequential(nn.Linear(in_dim,64),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(64, out_dim))
        # nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)


    def forward(self, input):
        transl_feat = self.layers(input)
        return transl_feat
