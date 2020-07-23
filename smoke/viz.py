import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image

class detectionInfo(object):
    def __init__(self, array):
        self.detection_class = array[0]

        # local orientation = alpha + pi/2
        self.alpha = float(array[1])

        # in pixel coordinate
        self.xmin = float(array[2])
        self.ymin = float(array[3])
        self.xmax = float(array[4])
        self.ymax = float(array[5])

        # height, weigh, length in object coordinate, meter
        self.h = float(array[6])
        self.w = float(array[7])
        self.l = float(array[8])

        # x, y, z in camera coordinate, meter
        self.tx = float(array[9])
        self.ty = float(array[10])
        self.tz = float(array[11])

        # global orientation [-pi, pi] along y axis
        self.rot_global = float(array[12]) 

        self.score = float(array[13]) 

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi


def compute_birdviewbox(detection_info: detectionInfo, shape, scale):
    h = np.array(detection_info.h * scale)
    w = np.array(detection_info.w * scale)
    l = np.array(detection_info.l * scale)
    x = np.array(detection_info.tx * scale)
    y = np.array(detection_info.ty * scale)
    z = np.array(detection_info.tz * scale)
    rot_y = np.array(detection_info.rot_global)

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0, :]))

def draw_birdeyes(ax2, detection_info: detectionInfo, shape):
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(detection_info, shape, scale)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

def compute_3Dbox(K_3x4: np.ndarray, detection_info: detectionInfo):

    R = np.array([[np.cos(detection_info.rot_global), 0, np.sin(detection_info.rot_global)],
                  [0, 1, 0],
                  [-np.sin(detection_info.rot_global), 0, np.cos(detection_info.rot_global)]])

    x_corners = [0, detection_info.l, detection_info.l, detection_info.l, detection_info.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, detection_info.h, detection_info.h, 0, 0, detection_info.h, detection_info.h]  # -h
    z_corners = [0, 0, 0, detection_info.w, detection_info.w, detection_info.w, detection_info.w, 0]  # -w/2

    x_corners = [i - detection_info.l / 2 for i in x_corners]
    y_corners = [i - detection_info.h for i in y_corners]
    z_corners = [i - detection_info.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([detection_info.tx, detection_info.ty, detection_info.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = K_3x4.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(ax, corners_2D, detection_info: detectionInfo, color):
    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLY
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)

def visualization(image: Image.Image, K_3x4: np.ndarray, detections: np.ndarray):
    _, n_target_variables = detections.shape
    assert n_target_variables == 14

    fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
    gs = GridSpec(1, 4)
    gs.update(wspace=0)  # set the spacing between axes.

    ax = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])

    shape = 900
    birdimage = np.zeros((shape, shape, 3), np.uint8)

    for detection in detections:
        detection_info = detectionInfo(detection)

        color = {
            0: "green",   # Car
            1: "yellow",  # Cyclist
            2: "cyan"     # Pedestrian           
        }.get(int(detection_info.detection_class))
        
        corners_2D = compute_3Dbox(K_3x4, detection_info)
        draw_3Dbox(ax, corners_2D, detection_info, color)
        draw_birdeyes(ax2, detection_info, shape)

    # visualize 3D bounding box
    ax.imshow(image)
    ax.set_xticks([]) #remove axis value
    ax.set_yticks([])

    # plot camera view range
    x1 = np.linspace(0, shape / 2)
    x2 = np.linspace(shape / 2, shape)
    ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

    # visualize bird eye view
    ax2.imshow(birdimage, origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])

    figure_bytesio = BytesIO()
    fig.savefig(figure_bytesio, dpi=fig.dpi, bbox_inches='tight', pad_inches=0, format="jpg") 
    figure_bytes = figure_bytesio.getvalue()
    
    return figure_bytes, corners_2D