import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.xnets.afx import Conv


class DownBlock3d(nn.Module):
    def __init__(self, inp, oup, k=3, p=1, g=1):
        super(DownBlock3d, self).__init__()
        self.conv = nn.Conv3d(inp, oup, k, padding=p, groups=g)
        self.norm = nn.BatchNorm3d(oup, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class UpBlock3d(nn.Module):
    def __init__(self, inp, oup, k=3, p=1, g=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(inp, oup, k, padding=p, groups=g)
        self.norm = nn.BatchNorm3d(oup, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        return F.relu(self.norm(self.conv(out)))


class Encoder(nn.Module):
    def __init__(self, block_expansion, inp, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(inp if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))), k=3, p=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    def __init__(self, block_expansion, inp, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, k=3, p=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + inp

        self.conv = nn.Conv3d(self.out_filters, self.out_filters, 3, padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)

        return F.relu(self.norm(self.conv(out)))


class Hourglass(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DMN(nn.Module):  # DenseMotionNetwork
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=True):
        super(DMN, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (compress + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, 7, padding=3)
        self.compress = nn.Conv3d(feature_channel, compress, 1)
        self.norm = nn.BatchNorm3d(compress, affine=True)
        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters * reshape_depth, 1, 7, padding=3)
        else:
            self.occlusion = None

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 4, 16, 64, 64)
        identity_grid = self.make_coordinate_grid((d, h, w), ref=kp_source)  # (16, 64, 64, 3)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)  # (1, 1, d=16, h=64, w=64, 3)
        coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)

        k = coordinate_grid.shape[1]

        driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 1, 3)  # (bs, num_kp, d, h, w, 3)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)  # (bs, 1+num_kp, d, h, w, 3)
        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1,
                                                                  1)  # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp + 1), -1, d, h, w)  # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), d, h, w, -1))  # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, d, h, w))  # (bs, num_kp+1, c, d, h, w)

        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]  # (d=16, h=64, w=64)
        gaussian_driving = self.kp2gaussian(kp_driving, spatial_size=spatial_size,
                                            kp_variance=0.01)  # (bs, num_kp, d, h, w)
        gaussian_source = self.kp2gaussian(kp_source, spatial_size=spatial_size,
                                           kp_variance=0.01)  # (bs, num_kp, d, h, w)
        heatmap = gaussian_driving - gaussian_source  # (bs, num_kp, d, h, w)

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(
            heatmap.type()).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)  # (bs, 1+num_kp, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 32, 16, 64, 64)

        feature = self.compress(feature)  # (bs, 4, 16, 64, 64)
        feature = self.norm(feature)  # (bs, 4, 16, 64, 64)
        feature = F.relu(feature)  # (bs, 4, 16, 64, 64)

        out_dict = dict()

        # 1. deform 3d feature
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, 1+num_kp, d, h, w, 3)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (bs, 1+num_kp, c=4, d=16, h=64, w=64)

        # 2. (bs, 1+num_kp, d, h, w)
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        input = torch.cat([heatmap, deformed_feature], dim=2)  # (bs, 1+num_kp, c=5, d=16, h=64, w=64)
        input = input.view(bs, -1, d, h, w)  # (bs, (1+num_kp)*c=105, d=16, h=64, w=64)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)  # (bs, 1+num_kp, d=16, h=64, w=64)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)  # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)  # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)  # (bs, 3, d, h, w)  mask take effect in this place
        deformation = deformation.permute(0, 2, 3, 4, 1)  # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape
            prediction_reshape = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # Bx1x64x64
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

    @staticmethod
    def make_coordinate_grid(spatial_size, ref, **kwargs):
        d, h, w = spatial_size
        x = torch.arange(w).type(ref.dtype).to(ref.device)
        y = torch.arange(h).type(ref.dtype).to(ref.device)
        z = torch.arange(d).type(ref.dtype).to(ref.device)

        # NOTE: must be right-down-in
        x = (2 * (x / (w - 1)) - 1)  # the x axis faces to the right
        y = (2 * (y / (h - 1)) - 1)  # the y axis faces to the bottom
        z = (2 * (z / (d - 1)) - 1)  # the z axis faces to the inner

        yy = y.view(1, -1, 1).repeat(d, 1, w)
        xx = x.view(1, 1, -1).repeat(d, h, 1)
        zz = z.view(-1, 1, 1).repeat(1, h, w)

        meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

        return meshed

    @staticmethod
    def kp2gaussian(kp, spatial_size, kp_variance):
        mean = kp

        coordinate_grid = DMN.make_coordinate_grid(spatial_size, mean)
        number_of_leading_dimensions = len(mean.shape) - 1
        shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
        coordinate_grid = coordinate_grid.view(*shape)
        repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
        coordinate_grid = coordinate_grid.repeat(*repeats)

        # Preprocess kp shape
        shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
        mean = mean.view(*shape)

        mean_sub = (coordinate_grid - mean)

        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

        return out


class WN(nn.Module):
    def __init__(self, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel,
                 estimate_occlusion_map=False, dense_motion_params=None, **kwargs):
        super(WN, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        if dense_motion_params is not None:
            self.dense_motion_network = DMN(num_kp=num_kp, feature_channel=reshape_channel,
                                            estimate_occlusion_map=estimate_occlusion_map, **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.third = Conv(max_features, block_expansion * (2 ** num_down_blocks), k=(3, 3), p=(1, 1), act=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks),
                                out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map

    def deform_input(self, inp, deformation):
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, feature_3d, kp_driving, kp_source):
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            dense_motion = self.dense_motion_network(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']  # Bx16x64x64x3
            out = self.deform_input(feature_3d, deformation)  # Bx32x16x64x64

            bs, c, d, h, w = out.shape  # Bx32x16x64x64
            out = out.view(bs, c * d, h, w)  # -> Bx512x64x64
            out = self.third(out)  # -> Bx256x64x64
            out = self.fourth(out)  # -> Bx256x64x64

            if self.flag_use_occlusion_map and (occlusion_map is not None):
                out = out * occlusion_map

        ret_dct = {
            'occlusion_map': occlusion_map,
            'deformation': deformation,
            'out': out,
        }

        return ret_dct
