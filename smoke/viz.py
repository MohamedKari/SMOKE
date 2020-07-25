import os
from io import BytesIO
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image

from .detection_info import DetectionInfo

def compute_birdviewbox(detection_info: DetectionInfo, shape, scale):
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

def draw_birdeyes(ax2, detection_info: DetectionInfo, shape):
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(detection_info, shape, scale)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

def compute_3Dbox(K_3x4: np.ndarray, detection_info: DetectionInfo):

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

def draw_3Dbox(ax, corners_2D, detection_info: DetectionInfo, color):
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

def visualization(image: Image.Image, K_3x4: np.ndarray, detection_infos: List[DetectionInfo]) -> Tuple[bytes, List[List[Tuple[float, float]]]]:

    # TODO: Refactor to PIL instead of matplotlib.
    fig1 = plt.figure(1, figsize=(20.00, 5.12), dpi=100)
    fig2 = plt.figure(2, figsize=(20.00, 5.12), dpi=100)

    shape = 900
    birdimage = np.zeros((shape, shape, 3), np.uint8)

    corners_2D_per_box = list()
    for detection_info in detection_infos:
        color = {
            0: "green",   # Car
            1: "yellow",  # Cyclist
            2: "cyan"     # Pedestrian           
        }.get(int(detection_info.detection_class))
        
        corners_2D = compute_3Dbox(K_3x4, detection_info)
        plt.figure(1)
        draw_3Dbox(plt.gca(), corners_2D, detection_info, color)
        plt.figure(2)
        draw_birdeyes(plt.gca(), detection_info, shape)

        corners_2D_per_box.append(corners_2D)

    # visualize 3D bounding box
    plt.figure(1)
    plt.gca().imshow(image)
    plt.gca().set_xticks([]) #remove axis value
    plt.gca().set_yticks([])

    # plot camera view range
    plt.figure(2)
    x1 = np.linspace(0, shape / 2)
    x2 = np.linspace(shape / 2, shape)
    plt.gca().plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
    plt.gca().plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
    plt.gca().plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

    # visualize bird eye view
    plt.gca().imshow(birdimage, origin='lower')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.figure(1)
    figure1_bytesio = BytesIO()
    fig1.savefig(figure1_bytesio, dpi=fig1.dpi, bbox_inches='tight', pad_inches=0, format="jpg") 
    figure1_bytes = figure1_bytesio.getvalue()

    plt.figure(2)
    figure2_bytesio = BytesIO()
    fig2.savefig(figure2_bytesio, dpi=fig2.dpi, bbox_inches='tight', pad_inches=0, format="jpg") 
    figure2_bytes = figure2_bytesio.getvalue()
    
    return figure1_bytes, figure2_bytes, corners_2D_per_box