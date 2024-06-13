import torch
import torch.nn as nn



class ResnetPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    Args:
        out_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, out_dim, hidden_dim):
        super().__init__()
        self.out_dim = out_dim

        dim = 3
        self.actvn = nn.ReLU()
        self.fc_pos_0 = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim, hidden_dim)
        # self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()

    @staticmethod
    def pool(x, dim=-1, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        # output size: B x T X F
        net = self.fc_pos_0(p)  # p: [bs, n_pts, 3], net: [bs, n_pts, 512]
        net = self.block_0(net)  # [bs, n_pts, 512]
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())  # [32, 1, 512] --> expand, [32, n_pts, 512]
        net = torch.cat([net, pooled], dim=2)  # [bs, n_pts, 1024]

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)  # [bs, n_pts, 1024]

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)  # [bs, n_pts, 1024]

        net = self.block_3(net)  # [bs, n_pts, 512]
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)
        #
        # net = self.block_4(net)

        # to  B x F
        net = self.pool(net, dim=1)  # [bs, hidden_dim]

        c = self.fc_c(self.act(net))  # [bs, out_dim]

        return c


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
