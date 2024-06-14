import torch
import torch.nn as nn



class SceneEncMLP(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    Args:
        out_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, in_dim, out_dim, hsize1=1024, hsize2=512):
        super().__init__()
        self.out_dim = out_dim

        self.actvn = nn.ReLU()
        self.fc_pos_0 = nn.Linear(in_dim, hsize1)
        self.block_0 = ResnetBlockFC(hsize1, hsize2, hsize2)
        self.block_1 = ResnetBlockFC(hsize2, hsize2, hsize2)
        self.block_2 = ResnetBlockFC(hsize2, hsize2, hsize2)
        self.block_3 = ResnetBlockFC(hsize2*3, hsize2, hsize2)
        # self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hsize2, out_dim)

        self.act = nn.ReLU()

    @staticmethod
    def pool(x, dim=-1, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        # p: [bs, 3, n_pts]
        net = self.fc_pos_0(p)  # , net: [bs, 3, 1024]
        net = self.block_0(net)  # [bs, 3, 512]
        net = self.block_1(net)  # [bs, 3, 512]
        net = self.block_2(net)  # [bs, 3, 512]
        net = net.reshape(net.shape[0], -1)  # [bs, 3*512]
        net = self.block_3(net)  # [bs, 512]
        # net = self.actvn(self.fc_c(net))  # [bs, out_dim]
        net = self.fc_c(self.actvn(net))  # [bs, out_dim]
        return net


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
