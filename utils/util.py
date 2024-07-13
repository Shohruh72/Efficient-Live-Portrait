import os
import cv2
import torch
import imageio
import numpy as np
import onnxruntime
from utils import crop
from glob import glob
from utils import crop
import torch.nn.functional as F
from rich.progress import track
from insightface import model_zoo
from utils.crop import _transform_pts
from numpy.linalg import norm as l2norm


# --------------------------------------------- helper functions -------------------------------------------------------


def dct2device(dct):
    for key in dct:
        dct[key] = torch.tensor(dct[key]).cuda()
    return dct


def dump(wfp, obj):
    import pickle
    wd = os.path.split(wfp)[0]
    if wd != "" and not os.path.exists(wd):
        mkdir(wd)

    _suffix = suffix(wfp)
    if _suffix == "npy":
        np.save(wfp, obj)
    elif _suffix == "pkl":
        pickle.dump(obj, open(wfp, "wb"))
    else:
        raise Exception("Unknown type: {}".format(_suffix))


def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def concat_feat(kp_source, kp_driving):
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src, bs_dri = kp_source.shape[0], kp_driving.shape[0]

    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def paste_back(mask_crop, crop_M_c2o, dsize):
    """prepare mask for later image paste back
    """
    mask_ori = crop._transform_img(mask_crop, crop_M_c2o, dsize)
    mask_ori = mask_ori.astype(np.float32) / 255.
    return mask_ori


def paste_back2(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image
    """
    dsize = (img_ori.shape[1], img_ori.shape[0])
    result = crop._transform_img(img_crop, M_c2o, dsize=dsize)
    result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
    return result


def load_img_online(obj, mode="bgr", **kwargs):
    max_dim = kwargs.get("max_dim", 1920)
    n = kwargs.get("n", 2)
    if isinstance(obj, str):
        if mode.lower() == "gray":
            img = cv2.imread(obj, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(obj, cv2.IMREAD_COLOR)
    else:
        img = obj

    # Resize image to satisfy constraints
    img = resize_to_limit(img, max_dim=max_dim, division=n)

    if mode.lower() == "bgr":
        return crop.contiguous(img)
    elif mode.lower() == "rgb":
        return crop.contiguous(img[..., ::-1])
    else:
        raise Exception(f"Unknown mode {mode}")


def resize_to_limit(img, max_dim=1920, division=2):
    """
    ajust the size of the image so that the maximum dimension does not exceed
    max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if 0 < max_dim < max(h, w):
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = max(division, 1)
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


# --------------------------------------------------- Poses ------------------------------------------------------------
def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        device = pred.device
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 97.5

        return degree

    return pred


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * np.pi
    yaw = yaw_ / 180 * np.pi
    roll = roll_ / 180 * np.pi

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).cuda()
    zeros = torch.zeros([bs, 1]).cuda()
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose


# --------------------------------------------------- Calculations -----------------------------------------------------
def calculate_distance_ratio(lmk, idx1, idx2, idx3, idx4, eps=1e-6):
    return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
            (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))


def calc_eye_close_ratio(lmk, target_eye_ratio=None):
    lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12)
    righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36)
    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)


def calc_lip_close_ratio(lmk):
    return calculate_distance_ratio(lmk, 90, 102, 48, 66)


# --------------------------------------------------- video processing -------------------------------------------------
def exec_cmd(cmd):
    import subprocess
    return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def load_image_rgb(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_fps(filepath, default_fps=25):
    try:
        fps = cv2.VideoCapture(filepath).get(cv2.CAP_PROP_FPS)

        if fps in (0, None):
            fps = default_fps
    except Exception as e:
        fps = default_fps

    return fps


def load_driving_info(driving_info):
    video_ori = []

    def load_images_from_directory(directory):
        image_paths = sorted(glob(os.path.join(directory, '*.png')) + glob(os.path.join(directory, '*.jpg')))
        return [load_image_rgb(im_path) for im_path in image_paths]

    def load_images_from_video(file_path):
        reader = imageio.get_reader(file_path, "ffmpeg")
        return [image for _, image in enumerate(reader)]

    if os.path.isdir(driving_info):
        video_ori = load_images_from_directory(driving_info)
    elif os.path.isfile(driving_info):
        video_ori = load_images_from_video(driving_info)

    return video_ori


def images2video(images, wfp, **kwargs):
    fps = kwargs.get('fps', 30)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    image_mode = kwargs.get('image_mode', 'rgb')
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

    writer = imageio.get_writer(
        wfp, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat,
        macro_block_size=macro_block_size
    )

    n = len(images)
    for i in track(range(n), description='Writing', transient=True):
        if image_mode.lower() == 'bgr':
            writer.append_data(images[i][..., ::-1])
        else:
            writer.append_data(images[i])

    writer.close()


def concat_frames(driving_image_lst, source_image, I_p_lst):
    # TODO: add more concat style, e.g., left-down corner driving
    out_lst = []
    h, w, _ = I_p_lst[0].shape

    for idx, _ in track(enumerate(I_p_lst), total=len(I_p_lst), description='Concatenating result...'):
        I_p = I_p_lst[idx]
        source_image_resized = cv2.resize(source_image, (w, h))

        if driving_image_lst is None:
            out = np.hstack((source_image_resized, I_p))
        else:
            driving_image = driving_image_lst[idx]
            driving_image_resized = cv2.resize(driving_image, (w, h))
            out = np.hstack((driving_image_resized, source_image_resized, I_p))

        out_lst.append(out)
    return out_lst


def add_audio_to_video(silent_video_path, audio_video_path, output_video_path):
    import subprocess
    cmd = [
        'ffmpeg',
        '-y',
        '-i', f'"{silent_video_path}"',
        '-i', f'"{audio_video_path}"',
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        f'"{output_video_path}"'
    ]

    try:
        exec_cmd(' '.join(cmd))
        print(f"Video with audio generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


# --------------------------------------------------------- Face Models-------------------------------------------------
class Face(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


class FaceAnalysis:
    def __init__(self, allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}

        onnx_files = ['./weights/insightface/2d106det.onnx', './weights/insightface/det_10g.onnx']
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                # print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        # print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 1)

        return dimg


class LandmarkRunner(object):
    def __init__(self, **kwargs):
        ckpt_path = kwargs.get('ckpt_path')

        self.session = onnxruntime.InferenceSession(ckpt_path, providers=[('CUDAExecutionProvider', {'device_id': 0})])

    def _run(self, inp):
        out = self.session.run(None, {'input': inp})
        return out

    def run(self, img_rgb, lmk=None):
        crop_dct = crop.crop_image(img_rgb, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
        img_crop_rgb = crop_dct['img_crop']

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        # 2d landmarks 203 points
        lmk = to_ndarray(out_pts[0]).reshape(-1, 2) * 224  # scale to 0-224
        lmk = _transform_pts(lmk, M=crop_dct['M_c2o'])

        return lmk

    def warmup(self):
        dummy_image = np.zeros((1, 3, 224, 224), dtype=np.float32)

        _ = self._run(dummy_image)


class FaceAnalysisDIY(FaceAnalysis):
    def __init__(self, allowed_modules=None, **kwargs):
        super().__init__(allowed_modules=allowed_modules, **kwargs)

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)  # the number of the detected faces, 0 means no limit
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # whether to do 106-point detection
        direction = kwargs.get('direction', 'large-small')  # sorting direction
        face_center = None

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue

                if (not flag_do_landmark_2d_106) and taskname == 'landmark_2d_106':
                    continue

                model.get(img_bgr, face)
            ret.append(face)

        ret = self.sort_by_direction(ret, direction, face_center)
        return ret

    def warmup(self):
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

    @staticmethod
    def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
        if len(faces) <= 0:
            return faces

        if direction == 'left-right':
            return sorted(faces, key=lambda face: face['bbox'][0])
        if direction == 'right-left':
            return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
        if direction == 'top-bottom':
            return sorted(faces, key=lambda face: face['bbox'][1])
        if direction == 'bottom-top':
            return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
        if direction == 'small-large':
            return sorted(faces,
                          key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
        if direction == 'large-small':
            return sorted(faces,
                          key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]),
                          reverse=True)
        if direction == 'distance-from-retarget-face':
            return sorted(faces, key=lambda face: (((face['bbox'][2] + face['bbox'][0]) / 2 - face_center[0]) ** 2 + (
                    (face['bbox'][3] + face['bbox'][1]) / 2 - face_center[1]) ** 2) ** 0.5)
        return faces
