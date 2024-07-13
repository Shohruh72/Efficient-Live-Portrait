import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, inp, oup, g=1, k=3, p=1, act=False, use_pool=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, k, padding=p, groups=g)
        self.norm = nn.BatchNorm2d(oup, affine=True)
        self.activation = nn.LeakyReLU() if act else nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class DownBlock2d(nn.Module):
    def __init__(self, inp, oup, k=3, p=1, g=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(inp, oup, k, padding=p, groups=g)
        self.norm = nn.BatchNorm2d(oup, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = F.relu(self.norm(self.conv(x)))
        out = self.pool(out)
        return out


class ResBlock3d(nn.Module):
    def __init__(self, inp, k, p):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(inp, inp, k, padding=p)
        self.conv2 = nn.Conv3d(inp, inp, k, padding=p)
        self.norm1 = nn.BatchNorm3d(inp, affine=True)
        self.norm2 = nn.BatchNorm3d(inp, affine=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return out + x


class AFX(nn.Module):
    def __init__(self, image_channel, block_expansion, num_down_blocks, max_features, reshape_channel, reshape_depth,
                 num_resblocks):
        super(AFX, self).__init__()
        self.image_channel = image_channel
        self.block_expansion = block_expansion
        self.num_down_blocks = num_down_blocks
        self.max_features = max_features
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.first = Conv(image_channel, block_expansion, k=(3, 3), p=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            inp = min(max_features, block_expansion * (2 ** i))
            oup = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(inp, oup, k=(3, 3), p=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(oup, max_features, 1, 1)

        self.resblocks_3d = nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, k=3, p=1))

    def forward(self, source_image):
        out = self.first(source_image)  # Bx3x256x256 -> Bx64x256x256

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape  # ->Bx512x64x64

        f_s = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)  # ->Bx32x16x64x64
        f_s = self.resblocks_3d(f_s)  # ->Bx32x16x64x64
        return f_s
