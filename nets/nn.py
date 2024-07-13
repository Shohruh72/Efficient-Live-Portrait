import os.path
import cv2
import yaml
import torch
import numpy as np
from utils import util
from utils.crop import Cropper
from rich.progress import track
from collections import OrderedDict
from nets.xnets import afx, mx, spad, wn, stitch


def remove_ddp_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


def load_model(weight, model_cfg, model_type):
    model_params = model_cfg['model_params'][f'{model_type}_params']

    if model_type == 'afx':
        model = afx.AFX(**model_params).cuda()
    elif model_type == 'mx':
        model = mx.MX(**model_params).cuda()
    elif model_type == 'wn':
        model = wn.WN(**model_params).cuda()
    elif model_type == 'spade':
        model = spad.SPDec(**model_params).cuda()
    elif model_type == 'stitch':
        config = model_cfg['model_params']['stitch_params']
        checkpoint = torch.load(weight, map_location=lambda storage, loc: storage)

        stitcher = stitch.StitchRNet(**config.get('stitching'))
        stitcher.load_state_dict(remove_ddp_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.cuda()
        stitcher.eval()

        lip_rt = stitch.StitchRNet(**config.get('lip'))
        lip_rt.load_state_dict(remove_ddp_key(checkpoint['retarget_mouth']))
        lip_rt = lip_rt.cuda()
        lip_rt.eval()

        eye_rt = stitch.StitchRNet(**config.get('eye'))
        eye_rt.load_state_dict(remove_ddp_key(checkpoint['retarget_eye']))
        eye_rt = eye_rt.cuda()
        eye_rt.eval()

        return {'stitching': stitcher, 'lip': lip_rt, 'eye': eye_rt}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(weight, map_location=lambda storage, loc: storage))
    return model.eval()


class LPW(object):  # LivePortraitWrapper
    def __init__(self, args):
        self.args = args
        model_cfg = yaml.load(open('utils/configs.yaml', 'r'), Loader=yaml.SafeLoader)

        self.afx_model = load_model('./weights/afx.pth', model_cfg, 'afx')
        self.mx_model = load_model('./weights/mx.pth', model_cfg, 'mx')
        self.wn_model = load_model('./weights/wn.pth', model_cfg, 'wn')
        self.spade_model = load_model('./weights/spad.pth', model_cfg, 'spade')
        if os.path.exists('weights/stitch.pth'):
            self.stitch_model = load_model('./weights/stitch.pth', model_cfg, 'stitch')
        else:
            self.stitch_model = None

    def prepare_source(self, img):
        h, w = img.shape[:2]
        if (h, w) != (self.args.input_size, self.args.input_size):
            image = cv2.resize(img, (self.args.input_size, self.args.input_size))
        else:
            image = img.copy()

        if image.ndim == 3:
            image = image[np.newaxis].astype(np.float32) / 255.
        elif image.ndim == 4:
            image = image.astype(np.float32) / 255.
        else:
            raise ValueError(f'img ndim should be 3 or 4: {image.ndim}')
        return torch.from_numpy(np.clip(image, 0, 1)).permute(0, 3, 1, 2).cuda()

    def prepare_video(self, image):
        if isinstance(image, list):
            _img = np.array(image)[..., np.newaxis]
        elif isinstance(image, np.ndarray):
            _img = image
        else:
            raise ValueError(f'image type error: {type(image)}')

        y = _img.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)

        return y.cuda()

    def extract_feature_3d(self, x):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                feature_3d = self.afx_model(x)

        return feature_3d.float()

    def get_kp_info(self, x, **kwargs):
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                kp_info = self.mx_model(x)

            for k, v in kp_info.items():
                if isinstance(v, torch.Tensor):
                    kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = util.headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
            kp_info['yaw'] = util.headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
            kp_info['roll'] = util.headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def transform_keypoint(self, kp_info):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        kp: BxNx3
        """
        kp = kp_info['kp']  # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = util.headpose_pred_to_degree(pitch)
        yaw = util.headpose_pred_to_degree(yaw)
        roll = util.headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = util.get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    def retarget_eye(self, kp_source, eye_close_ratio):
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp+2)
        """
        feat_eye = util.concat_feat(kp_source, eye_close_ratio)

        with torch.no_grad():
            delta = self.stitch_model['eye'](feat_eye)

        return delta

    def retarget_lip(self, kp_source, lip_close_ratio):
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        feat_lip = util.concat_feat(kp_source, lip_close_ratio)

        with torch.no_grad():
            delta = self.stitch_model['lip'](feat_lip)

        return delta

    def stitch(self, kp_source, kp_driving):
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        feat_stiching = util.concat_feat(kp_source, kp_driving)

        with torch.no_grad():
            delta = self.stitch_model['stitching'](feat_stiching)

        return delta

    def stitching(self, kp_source, kp_driving):
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        if self.stitch_model is not None:
            bs, num_kp = kp_source.shape[:2]

            kp_driving_new = kp_driving.clone()
            delta = self.stitch(kp_source, kp_driving_new)

            delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty

            return kp_driving_new

        return kp_driving

    def warp_decode(self, feature_3d, kp_source, kp_driving):
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """
        # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                # get decoder input
                ret_dct = self.wn_model(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
                # decode
                ret_dct['out'] = self.spade_model(feature=ret_dct['out'])

            # float the dict
            for k, v in ret_dct.items():
                if isinstance(v, torch.Tensor):
                    ret_dct[k] = v.float()

        return ret_dct

    def parse_output(self, out):
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    def calc_driving_ratio(self, driving_lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in driving_lmk_lst:
            input_eye_ratio_lst.append(util.calc_eye_close_ratio(lmk[None]))
            input_lip_ratio_lst.append(util.calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = util.calc_eye_close_ratio(source_lmk[None])
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().cuda()
        c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).cuda()
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = util.calc_lip_close_ratio(source_lmk[None])
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().cuda()
        c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).cuda().reshape(1, 1)  # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1)  # 1x2
        return combined_lip_ratio_tensor


class LPP(object):  # LivePortraitPipeline

    def __init__(self, args, flag=False, audio_flag=False):
        self.args = args
        self.flag = flag
        self.audio_flag = audio_flag
        self.cropper = Cropper()
        self.lpw = LPW(args)

    def execute(self):
        img_rgb = util.load_image_rgb(self.args.input_image)
        img_rgb = util.resize_to_limit(img_rgb, 1280, 2)

        crop_info = self.cropper.crop_source_image(img_rgb)
        if crop_info is None:
            raise Exception("No face detected in the source image!")
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        I_s = self.lpw.prepare_source(img_crop_256x256)

        x_s_info = self.lpw.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = util.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.lpw.extract_feature_3d(I_s)
        x_s = self.lpw.transform_keypoint(x_s_info)

        # let lip-open scalar to be 0 at first
        flag_lip_zero = True
        if flag_lip_zero:
            c_d_lip_pre_anim = [0.]
            lip_ratio_pre_anim = self.lpw.calc_combined_lip_ratio(c_d_lip_pre_anim, source_lmk)

            if lip_ratio_pre_anim[0][0] < 0.03:
                flag_lip_zero = False
            else:
                lip_delta_pre_anim = self.lpw.retarget_lip(x_s, lip_ratio_pre_anim)

        output_fps = int(util.get_fps(self.args.input_video))

        print(f"Load video file (mp4 mov avi etc...): {self.args.input_video}")
        driving_rgb_lst = util.load_driving_info(self.args.input_video)

        print("----------------------------------Start making motion template...----------------------------------")
        if self.flag:
            ret = self.cropper.crop_driving_video(driving_rgb_lst)
            print(f'Driving video is cropped, {len(ret["frame_crop_lst"])} frames are processed.')
            driving_rgb_crop_lst, driving_lmk_crop_lst = ret['frame_crop_lst'], ret['lmk_crop_lst']
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
        else:
            driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in
                                            driving_rgb_lst]  # force to resize to 256x256

        c_d_eyes_lst, c_d_lip_lst = self.lpw.calc_driving_ratio(driving_lmk_crop_lst)
        I_d_lst = self.lpw.prepare_video(driving_rgb_crop_256x256_lst)
        template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

        wfp_template = os.path.join(os.path.dirname(self.args.input_video),
                                    util.prefix(os.path.basename(self.args.input_video))) + '.pkl'

        n_frames = I_d_lst.shape[0]
        util.dump(wfp_template, template_dct)

        mask_crop = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '../demo/inputs/mask_template.png'), cv2.IMREAD_COLOR)
        mask_ori_float = util.paste_back(mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))

        I_p_lst = []
        I_p_pstbk_lst = []
        R_d_0, x_d_0_info = None, None

        for i in track(range(n_frames), description='🚀Animating...', total=n_frames):
            x_d_i_info = template_dct['motion'][i]
            x_d_i_info = util.dct2device(x_d_i_info)
            R_d_i = x_d_i_info['R_d']

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
            scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
            t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if flag_lip_zero:
                x_d_i_new = (self.lpw.stitching(x_s, x_d_i_new) +
                                    lip_delta_pre_anim.reshape(-1, x_s.shape[1], 3))
            else:
                x_d_i_new = self.lpw.stitching(x_s, x_d_i_new)

            out = self.lpw.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.lpw.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            # TODO: pasteback is slow, considering optimize it using multi-threading or GPU
            I_p_pstbk = util.paste_back2(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
            I_p_pstbk_lst.append(I_p_pstbk)

        # driving frame | source image | generation, or source image | generation
        frames_concatenated = util.concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256, I_p_lst)
        wfp_concat = os.path.join(self.args.output_dir,
                                  f'{util.prefix(os.path.basename(self.args.input_image))}--'
                                  f'{util.prefix(os.path.basename(self.args.input_video))}_concat.mp4')

        util.images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

        # save drived result
        wfp = os.path.join(self.args.output_dir,
                           f'{util.prefix(os.path.basename(self.args.input_image))}--'
                           f'{util.prefix(os.path.basename(self.args.input_video))}.mp4')

        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            util.images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
        else:
            util.images2video(I_p_lst, wfp=wfp, fps=output_fps)

        if self.audio_flag:
            wfp_with_audio = os.path.join(self.args.output_dir,
                                          f'{util.prefix(os.path.basename(self.args.input_image))}--'
                                          f'{util.prefix(os.path.basename(self.args.input_video))}_with_audio.mp4')
            util.add_audio_to_video(wfp, self.args.input_video, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)

        # final log
        if wfp_template not in (None, ''):
            print(f'Animated template: {wfp_template}, you can specify `-d` argument with this template path next time '
                  f'to avoid cropping video, motion making and protecting privacy.')
        print(f'Animated video: {wfp}')
        print(f'Animated video with concact: {wfp_concat}')
        return wfp, wfp_concat

    def make_motion_template(self, I_d_lst, c_d_eyes_lst, c_d_lip_lst, **kwargs):
        n_frames = I_d_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_d_eyes_lst': [],
            'c_d_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s_d, R_d, δ_d and t_d for inference
            I_d_i = I_d_lst[i]
            x_d_i_info = self.lpw.get_kp_info(I_d_i)
            R_d_i = util.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

            item_dct = {
                'scale': x_d_i_info['scale'].cpu().numpy().astype(np.float32),
                'R_d': R_d_i.cpu().numpy().astype(np.float32),
                'exp': x_d_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_d_i_info['t'].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_d_eyes = c_d_eyes_lst[i].astype(np.float32)
            template_dct['c_d_eyes_lst'].append(c_d_eyes)

            c_d_lip = c_d_lip_lst[i].astype(np.float32)
            template_dct['c_d_lip_lst'].append(c_d_lip)

        return template_dct
