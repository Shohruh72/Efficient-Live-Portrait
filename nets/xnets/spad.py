import torch
import torch.nn as nn
import torch.nn.functional as F


class SP(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        n_hidden = 128
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, n_hidden, 3, padding=1), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(n_hidden, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(n_hidden, norm_nc, 3, padding=1)

    def forward(self, x, seg_map):
        act = self.mlp_shared(F.interpolate(seg_map, size=x.size()[2:], mode='nearest'))
        return self.param_free_norm(x) * (1 + self.mlp_gamma(act)) + self.mlp_beta(act)


class SPConv(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        self.use_se = use_se
        f_middle = min(fin, fout)
        self.learned_shortcut = (fin != fout)

        self.conv_0 = nn.Conv2d(fin, f_middle, 3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(f_middle, fout, 3, padding=dilation, dilation=dilation)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)
        if 'spectral' in norm_G:
            self.conv_0 = nn.utils.spectral_norm(self.conv_0)
            self.conv_1 = nn.utils.spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = nn.utils.spectral_norm(self.conv_s)

        self.norm_0 = SP(fin, label_nc)
        self.norm_1 = SP(f_middle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SP(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPDec(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):
            inp = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'

        self.fc = nn.Conv2d(inp, 2 * inp, 3, padding=1)
        self.G_middle_0 = SPConv(2 * inp, 2 * inp, norm_G, inp)
        self.G_middle_1 = SPConv(2 * inp, 2 * inp, norm_G, inp)
        self.G_middle_2 = SPConv(2 * inp, 2 * inp, norm_G, inp)
        self.G_middle_3 = SPConv(2 * inp, 2 * inp, norm_G, inp)
        self.G_middle_4 = SPConv(2 * inp, 2 * inp, norm_G, inp)
        self.G_middle_5 = SPConv(2 * inp, 2 * inp, norm_G, inp)
        self.up_0 = SPConv(2 * inp, inp, norm_G, inp)
        self.up_1 = SPConv(inp, out_channels, norm_G, inp)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                                          nn.PixelShuffle(upscale_factor=2))

    def forward(self, feature):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x
