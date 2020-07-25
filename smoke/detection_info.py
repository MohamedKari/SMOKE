from dataclasses import dataclass

import numpy as np

@dataclass
class DetectionInfo():
    
    # 0: Car, 1: Cyclist, 2: Pedestrian
    detection_class: int
    
    # local orientation = alpha + pi/2
    alpha: float
    
    # 2D Bounding Box in pixel coordinate
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    # Dimensions (height, weigh, length) in object coordinates [meter]
    h: float
    w: float
    l: float

    # Translation (x, y, z) in camera coordinates [meter]
    tx: float
    ty: float
    tz: float

    # global orientation [-pi, pi] along y axis
    rot_global: float

    # Prediction score [0, 1]
    score: float

    @staticmethod
    def from_array(array):

        return DetectionInfo(
            detection_class=int(array[0]),
            alpha=float(array[1]),
            xmin=float(array[2]),
            ymin=float(array[3]),
            xmax=float(array[4]),
            ymax=float(array[5]),
            h=float(array[6]),
            w=float(array[7]),
            l=float(array[8]),
            tx=float(array[9]),
            ty=float(array[10]),
            tz=float(array[11]),
            rot_global=float(array[12]),
            score=float(array[13])
        )

    def detection_class_string(self) -> str:
        return {
            0: "Car",
            1: "Cyclist",
            2: "Pedestrian"
        }.get(self.detection_class)

def get_camera_intrinsics(image_height, image_width):
    offset_x = 12 # ???
    offset_y = 15 # ???

    required_image_height = 375
    dependent_image_width = image_width * required_image_height / image_height

    alpha = 721.
    u = (dependent_image_width / 2 - offset_x)
    v = (required_image_height / 2 - offset_y)
    
    _K_3x4 = np.array(
        [
            # kitti
            #[721.5377,  0.,         609.5593,   44.85728],      # pylint: disable=bad-whitespace
            #[0.,        721.5377,   172.854,    .2163791],      # pylint: disable=bad-whitespace
            #[0.,        0.,         1.,         .002745884]     # pylint: disable=bad-whitespace
        
            # kitti (image height 375) but variable width
            [alpha,     0.,         u,          .0], 
            [0.,        alpha,      v,          .0], 
            [0.,        0.,         1.,         .0]
        ], 
        
        dtype=np.float32
    )

    return _K_3x4