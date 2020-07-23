from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .viz import visualization

from smoke.config import cfg
from smoke.data.transforms import build_transforms
from smoke.engine import default_setup
from smoke.modeling.detector import build_detection_model
from smoke.modeling.heatmap_coder import get_transfrom_matrix
from smoke.structures.image_list import to_image_list
from smoke.structures.params_3d import ParamsList
from smoke.utils.check_point import DetectronCheckpointer

@dataclass
class Detection():
    
    detection_class: str
    
    # Trash
    trash_1: str # truncated
    trash_2: str # occluded
    
    # ?
    alpha: float

    # 2D Box
    box2d_xmin: float
    box2d_ymin: float
    box2d_xmax: float
    box2d_ymax: float

    # Dimensions
    height: float
    width: float
    length: float

    # Location
    location_x: float
    location_y: float
    location_z: float

    # Rotation around y dimension
    ry: float
    
    # Detection Score
    threshold: float

@dataclass
class ProgrammaticArgs():
    config_file = "configs/smoke_gn_vector.yaml"
    eval_only = True
    num_gpus = 1

class SmokeSession():
    def __init__(self, K_3x3: np.ndarray):
        self.K_3x3 = K_3x3

        args = ProgrammaticArgs()

        cfg.merge_from_file(args.config_file)
        cfg.freeze()
        default_setup(cfg, args)

        self.model = build_detection_model(cfg)
        self.model.to(cfg.MODEL.DEVICE)

        checkpointer = DetectronCheckpointer(
            cfg, self.model, save_dir=cfg.OUTPUT_DIR
        )
        _ = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=True)

        self.model.eval()

    def detect(self, frame_id: int, frame_image: Image.Image) -> dict:

        # KITTI specifics
        width = 1242.
        height = 375.

        # Model specifics
        input_width = cfg.INPUT.WIDTH_TRAIN
        input_height = cfg.INPUT.HEIGHT_TRAIN
        output_width = input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        output_height = input_height // cfg.MODEL.BACKBONE.DOWN_RATIO

        # smoke/data/datasets/kitti.py#KITTIDataset.__get_item__
        center = np.array([i / 2 for i in frame_image.size], dtype=np.float32)
        size = np.array([i for i in frame_image.size], dtype=np.float32)

        assert size[0] == width
        assert size[1] == height

        center_size = [center, size]
        
        trans_affine = get_transfrom_matrix(
            center_size,
            [input_width, input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        frame_image = frame_image.transform(
            (input_width, input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [output_width, output_height]
        )

        target = ParamsList(image_size=size, is_train=False)
        target.add_field("trans_mat", trans_mat)
        target.add_field("K", self.K_3x3)

        transforms = build_transforms(cfg, is_train=False)
        frame_tensor, target = transforms(frame_image, target)

        frame_tensors, targets = to_image_list(frame_tensor), (target, )

        frame_tensors = frame_tensors.to(cfg.MODEL.DEVICE)
        
        with torch.no_grad():
            output = self.model(frame_tensors, targets)
            output = output.to("cpu")

        return output

class VideoInference():
    pass


def get_camera_intrinsics():
    _K_3x4 = np.array(
        [
            [721.5377,  0.,         609.5593,   44.85728],      # pylint: disable=bad-whitespace
            [0.,        721.5377,   172.854,    .2163791],      # pylint: disable=bad-whitespace
            [0.,        0.,         1.,         .002745884]     # pylint: disable=bad-whitespace
        ], 
        dtype=np.float32
    )

    return _K_3x4

K_3x4 = get_camera_intrinsics()
K_3x3 = K_3x4[:3, :3]

smoke_session = SmokeSession(K_3x3)

image_path = "/app/datasets/kitti/testing/image_2/000146.png"
image = Image.open(image_path)

detection_output = smoke_session.detect(0, image)
figure_bytes, corners_2D = visualization(image, K_3x4, detection_output)
print("corners_2D of type", type(corners_2D), ":", corners_2D)

from pathlib import Path
Path("/app/tools/logs/inference/kitti_test/image.jpg").write_bytes(figure_bytes)


# RPC client & server
# On-the-fly resizing