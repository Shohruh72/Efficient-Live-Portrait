import cv2
import numpy as np
from utils import util
from typing import List, Tuple, Union
from math import sin, cos, acos, degrees
from dataclasses import dataclass, field


# ---------------------------------------------- Parsers ---------------------------------------------------------------
def parse_pt2_from_pt101(pt101, use_lip=True):
    """
    parsing the 2 points according to the 101 points, which cancels the roll
    """
    # the former version use the eye center, but it is not robust, now use interpolation
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt106(pt106, use_lip=True):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt203(pt203, use_lip=True):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)  # right eye center
    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt68(pt68, use_lip=True):
    """
    parsing the 2 points according to the 68 points, which cancels the roll
    """
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    if use_lip:
        pt5 = np.stack([
            np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
            pt68[lm_idx[0], :],  # nose
            pt68[lm_idx[5], :],  # lip
            pt68[lm_idx[6], :]  # lip
        ], axis=0)

        pt2 = np.stack([
            (pt5[0] + pt5[1]) / 2,
            (pt5[3] + pt5[4]) / 2
        ], axis=0)
    else:
        pt2 = np.stack([
            np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
        ], axis=0)

    return pt2


def parse_pt2_from_pt5(pt5, use_lip=True):
    """
    parsing the 2 points according to the 5 points, which cancels the roll
    """
    if use_lip:
        pt2 = np.stack([
            (pt5[0] + pt5[1]) / 2,
            (pt5[3] + pt5[4]) / 2
        ], axis=0)
    else:
        pt2 = np.stack([
            pt5[0],
            pt5[1]
        ], axis=0)
    return pt2


def parse_pt2_from_pt_x(pts, use_lip=True):
    if pts.shape[0] == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    elif pts.shape[0] == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)
    elif pts.shape[0] == 68:
        pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
    elif pts.shape[0] == 5:
        pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
    elif pts.shape[0] == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip)
    elif pts.shape[0] > 101:
        # take the first 101 points
        pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
    else:
        raise Exception(f'Unknow shape: {pts.shape}')

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2


# -------------------------------------------- Crop Helpers ------------------------------------------------------------

def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj


def _transform_pts(pts, M):
    """ conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]


def average_bbox_lst(bbox_lst):
    if len(bbox_lst) == 0:
        return None
    bbox_arr = np.array(bbox_lst)
    return np.mean(bbox_arr, axis=0).tolist()


def crop_image(img, pts, **kwargs):
    dsize = kwargs.get('dsize', 224)
    scale = kwargs.get('scale', 1.5)  # 1.5 | 1.6
    vy_ratio = kwargs.get('vy_ratio', -0.1)  # -0.0625 | -0.1

    m_inv, _ = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get('flag_do_rot', True),
    )

    img_crop = _transform_img(img, m_inv, dsize)  # origin to crop
    pt_crop = _transform_pts(pts, m_inv)

    m_o2c = np.vstack([m_inv, np.array([0, 0, 1], dtype=np.float32)])
    m_c2o = np.linalg.inv(m_o2c)

    ret_dct = {
        'M_o2c': m_o2c,  # from the original image to the cropped image 3x3
        'M_c2o': m_c2o,  # from the cropped image to the original image 3x3
        'img_crop': img_crop,  # the cropped image
        'pt_crop': pt_crop,  # the landmarks of the cropped image
    }

    return ret_dct


def parse_bbox_from_landmark(pts, **kwargs):
    center, size, angle = parse_rect_from_landmark(pts, **kwargs)
    cx, cy = center
    w, h = size

    # calculate the vertex positions before rotation
    bbox = np.array([
        [cx - w / 2, cy - h / 2],  # left, top
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],  # right, bottom
        [cx - w / 2, cy + h / 2]
    ], dtype=np.float32)

    # construct rotation matrix
    bbox_rot = bbox.copy()
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=np.float32)

    bbox_rot = (bbox_rot - center) @ R.T + center

    return {
        'center': center,  # 2x1
        'size': size,  # scalar
        'angle': angle,  # rad, counterclockwise
        'bbox': bbox,  # 4x2
        'bbox_rot': bbox_rot,  # 4x2
    }


def _transform_img(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=None):
    """ conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode, borderValue=(0, 0, 0))
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def crop_image_by_bbox(img, bbox, lmk=None, dsize=512, angle=None, flag_rot=False, **kwargs):
    left, top, right, bot = bbox
    if int(right - left) != int(bot - top):
        print(f'right-left {right - left} != bot-top {bot - top}')
    size = right - left

    src_center = np.array([(left + right) / 2, (top + bot) / 2], dtype=np.float32)
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)

    s = dsize / size  # scale
    if flag_rot and angle is not None:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = src_center[0], src_center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_o2c = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=np.float32
        )
    else:
        M_o2c = np.array(
            [[s, 0, tgt_center[0] - s * src_center[0]],
             [0, s, tgt_center[1] - s * src_center[1]]],
            dtype=np.float32
        )

    # if flag_rot and angle is None:
    # print('angle is None, but flag_rotate is True', style="bold yellow")

    img_crop = _transform_img(img, M_o2c, dsize=dsize, borderMode=kwargs.get('borderMode', None))
    lmk_crop = _transform_pts(lmk, M_o2c) if lmk is not None else None

    M_o2c = np.vstack([M_o2c, np.array([0, 0, 1], dtype=np.float32)])
    M_c2o = np.linalg.inv(M_o2c)

    # cv2.imwrite('crop.jpg', img_crop)

    return {
        'img_crop': img_crop,
        'lmk_crop': lmk_crop,
        'M_o2c': M_o2c,
        'M_c2o': M_c2o,
    }


def _estimate_similar_transform_from_pts(pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1, **kwargs):
    """ calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
        use_lip=kwargs.get('use_lip', True)
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)

    costheta, sintheta = cos(angle), sin(angle)
    cx, cy = center[0], center[1]  # ori center
    tcx, tcy = tgt_center[0], tgt_center[1]  # target center
    # need to infer
    M_INV = np.array(
        [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
         [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
        dtype=np.float32
    )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    return M_INV, M[:2, ...]


def parse_rect_from_landmark(pts, scale=1.5, need_square=True, vx_ratio=0, vy_ratio=0, use_deg_flag=False, **kwargs):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get('use_lip', True))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=np.float32)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    # print(uy)
    # print(ux)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = center0 + ux * center1[0] + uy * center1[1]  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = center + ux * (vx_ratio * size) + uy * \
             (vy_ratio * size)  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle


# -------------------------------------------- Cropper ------------------------------------------------------------
@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list

    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list

    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        flag_force_cpu = kwargs.get("flag_force_cpu", False)

        face_analysis = ["CUDAExecutionProvider"]
        self.landmark_runner = util.LandmarkRunner(ckpt_path='./weights/landmark.onnx')
        self.landmark_runner.warmup()

        self.face_analysis_wrapper = util.FaceAnalysisDIY(providers=face_analysis, )
        self.face_analysis_wrapper.prepare(ctx_id=0, det_size=(512, 512))
        self.face_analysis_wrapper.warmup()

    def crop_source_image(self, img_rgb_):
        img_rgb = img_rgb_.copy()  # copy it

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        src_face = self.face_analysis_wrapper.get(img_bgr, flag_do_landmark_2d_106=True, direction="large-small",
                                                  max_face_num=0, )

        if len(src_face) == 0:
            print("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            print(f"More than one face detected in the image, only pick one face by rule.")

        # NOTE: temporarily only pick the first face, to support multiple face in the future
        src_face = src_face[0]
        lmk = src_face.landmark_2d_106  # this is the 106 landmarks from insightface

        ret_dct = crop_image(img_rgb, lmk, dsize=512, scale=2.5, vx_ratio=0, vy_ratio=-0.125)
        lmk = self.landmark_runner.run(img_rgb, lmk)
        ret_dct["lmk_crop"] = lmk

        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / 512

        return ret_dct

    def crop_driving_video(self, driving_rgb_lst, **kwargs):
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")
        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    print(
                        f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
            ret_bbox = parse_bbox_from_landmark(lmk, scale=2.2, vx_ratio_crop_video=0.0, vy_ratio=-0.1, )["bbox"]
            bbox = [ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1], ]  # 4,
            trajectory.bbox_lst.append(bbox)  # bbox
            trajectory.frame_rgb_lst.append(frame_rgb)

        global_bbox = average_bbox_lst(trajectory.bbox_lst)

        for idx, (frame_rgb, lmk) in enumerate(zip(trajectory.frame_rgb_lst, trajectory.lmk_lst)):
            ret_dct = crop_image_by_bbox(frame_rgb, global_bbox, lmk=lmk, dsize=kwargs.get("dsize", 512),
                                         flag_rot=False, borderValue=(0, 0, 0), )
            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop"])

        return {"frame_crop_lst": trajectory.frame_rgb_crop_lst, "lmk_crop_lst": trajectory.lmk_crop_lst}

    def calc_lmks_from_cropped_video(self, driving_rgb_crop_lst, **kwargs):
        """Tracking based landmarks/alignment"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")

        for idx, frame_rgb_crop in enumerate(driving_rgb_crop_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(contiguous(frame_rgb_crop[..., ::-1]),
                                                          flag_do_landmark_2d_106=True, direction=direction)
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    raise Exception(f"No face detected in the frame #{idx}")
                elif len(src_face) > 1:
                    print(
                        f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.landmark_runner.run(frame_rgb_crop, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.landmark_runner.run(frame_rgb_crop, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
        return trajectory.lmk_lst
