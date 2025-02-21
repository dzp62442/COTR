import os
import setproctitle
import mmcv
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 17
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SPTIAL_SHAPE = [200, 200, 16]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]


colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],  # 4 car  Crimson
        [233, 150, 70, 255],   # 5 cons. Veh  Orangered
        [255, 61, 99, 255],  # 6 motorcycle  Darkorange
        [0, 0, 230, 255], # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone  Red
        [255, 140, 0, 255],# 9 trailer  Slategrey
        [255, 99, 71, 255],# 10 truck Burlywood
        [0, 207, 191, 255],    # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegeyation
], dtype=np.float32)

val_scene = ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
val_night = ['scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 
            'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 
            'scene-1067', 'scene-1068', 'scene-1069', 'scene-1070', 
            'scene-1071', 'scene-1072', 'scene-1073']

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4)
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # 自车直角坐标系

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    # 绘制自车点云
    # ego_pcd = o3d.geometry.PointCloud()
    # ego_points = generate_the_ego_car()
    # ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    # vis.add_geometry(ego_pcd)

    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,  # 是否显示体素的颜色，False 为纯黑
        points_colors=pcds_colors,
        voxelize=True,  # 是否显示体素的网格线
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,  # 是否显示大体素，True 方格体素，False 点云
        voxel_size=0.4
    )
    return vis


def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


def draw_frame(vis, view_control, look_at, front, up, zoom):
    """
    指定视角并渲染一帧

    Args:
        vis: Visualizer
    """

    view_control.set_lookat(look_at)
    view_control.set_front(front)
    view_control.set_up(up)
    view_control.set_zoom(zoom)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.line_width = 5

    vis.poll_events()
    vis.update_renderer()
    vis.run()

    occ_canvas = vis.capture_screen_float_buffer(do_render=True)
    occ_canvas = np.asarray(occ_canvas)
    occ_canvas = (occ_canvas * 255).astype(np.uint8)
    occ_canvas = occ_canvas[..., [2, 1, 0]]

    return occ_canvas

def draw(vis, imgs, out_dir, save_format, cam_positions=None, focal_positions=None, cam_names=None, canva_size=1000, scale_factor=4, mode='pred'):
    """
    根据 open3d Visualizer 渲染指定视角的图像并保存

    Args:
        vis: Visualizer
        save_format: args.format, image or video
        cam_positions: (Dx, Dy, Dz), bool
        focal_positions: [0.4, 0.4, 0.4]
        cam_names: Visualizer
        mode: pred or gt
    """

    view_control = vis.get_view_control()

    #! 渲染常规视角图像并保存
    look_at = np.array([-0.185, 0.513, 3.485])
    front = np.array([-0.974, -0.055, 0.221])
    up = np.array([0.221, 0.014, 0.975])
    zoom = np.array([0.08])

    normal_frame = draw_frame(vis, view_control, look_at, front, up, zoom)
    normal_frame_resize = cv2.resize(normal_frame, (canva_size, canva_size), interpolation=cv2.INTER_CUBIC)

    overall_img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
    overall_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate([imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1)
    overall_img[900 + canva_size * scale_factor:, :, :] = img_back
    overall_img = cv2.resize(overall_img, (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canva_size)))
    w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    overall_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
            w_begin:w_begin + canva_size, :] = normal_frame_resize

    if save_format == 'image':
        mmcv.mkdir_or_exist(out_dir)
        cv2.imwrite(os.path.join(out_dir, mode+'_normal.png'), normal_frame)
        cv2.imwrite(os.path.join(out_dir, mode+'_overall.png'), overall_img)
    
    #! 渲染鸟瞰视角图像并保存
    if save_format == 'image':
        look_at = np.array([0.75131739,  0.78265103, 92.21378558])
        front = np.array([0.75131739,  0.78265103, 93.21378558])
        up = np.array([0., 1., 0.])
        zoom = np.array([0.01])

        bev_frame = draw_frame(vis, view_control, look_at, front, up, zoom)
        cv2.imwrite(os.path.join(out_dir, mode+'_bev.png'), bev_frame)
    
    #! 渲染六相机分立图像并保存
    if save_format == 'image':
        cam_frames = []
        for i, cam_name in enumerate(cam_names):
            cam_position = cam_positions[i]
            focal_position = focal_positions[i]
            look_at = focal_position
            front = cam_position
            up = np.array([0., 0., 1.])
            zoom = np.array([0.5]) if i!=3 else np.array([0.8])  # 后相机视场角不同
            cam_frame = draw_frame(vis, view_control, look_at, front, up, zoom)
            cam_frames.append(cam_frame)

        img_size = imgs[0].shape  # 获取第一个原始图像的大小

        canva_size = 100  # 假设 canvas 尺寸
        scale_factor = 1  # 假设没有缩放
        split_img = np.zeros((img_size[0] * 4, img_size[1] * 3, 3), dtype=np.uint8)  # 创建一个空的大图，4行3列
        split_img[:img_size[0], :, :] = np.concatenate(imgs[:3], axis=1)  # 将前 3 张原始图像拼接到大图的第一行    
        img_back = np.concatenate([imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1)
        split_img[img_size[0] * 2: img_size[0] * 2 + img_size[0], :, :] = img_back  # 将后 3 张原始图像反转并拼接到大图的第三行

        # 将 cam_frames 中的每张渲染图像调整为原图像大小，并拼接到大图的第二行和第四行
        for i in range(3):
            occ_resized = cv2.resize(cam_frames[i], (img_size[1], img_size[0]))  # 调整 occ 图像尺寸
            split_img[img_size[0]:img_size[0] * 2, i * img_size[1]:(i + 1) * img_size[1], :] = occ_resized
        for i in range(3, 6):
            occ_resized = cv2.resize(cam_frames[i], (img_size[1], img_size[0]))  # 调整 occ 图像尺寸
            split_img[img_size[0] * 3:img_size[0] * 4, (i - 3) * img_size[1]:(i - 2) * img_size[1], :] = occ_resized

        cv2.imwrite(os.path.join(out_dir, mode+'_split.png'), split_img)
    
    return split_img


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument('--scene-idx', type=int, default=None, nargs='+', 
                        help='idx of scene to visualize, scene idx must in the val scene list.')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=1,
        help='Max number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load predicted results
    results_dir = args.res

    # load dataset information
    info_path = args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('Start visualizing results !')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # 遍历每个场景信息
    vis_number = 0  # 已进行可视化的帧数
    for cnt, info in enumerate(dataset['infos'][:len(dataset['infos'])]):
        scene_name = info['scene_name']
        sample_token = info['token']

        # 若设置了 --scene-idx 参数，则只对指定的场景进行可视化
        if args.scene_idx is not None and int(scene_name[6:]) not in args.scene_idx:
            continue

        vis_number += 1
        if vis_number > args.vis_frames:
            break
        if vis_number % 10 == 0:
            print('Visualize %d/%d' % (vis_number, min(args.vis_frames, len(dataset['infos']))))

        pred_occ_path = os.path.join(results_dir, scene_name, sample_token, 'pred.npz')
        gt_occ_path = info['occ_path']

        pred_occ = np.load(pred_occ_path)['pred']
        gt_data = np.load(os.path.join(gt_occ_path, 'labels.npz'))
        voxel_label = gt_data['semantics']
        lidar_mask = gt_data['mask_lidar']
        camera_mask = gt_data['mask_camera']

        # load imgs
        imgs = []
        for view in views:
            img = cv2.imread(info['cams'][view]['data_path'])
            imgs.append(img)

        # 解析 nuscenes 相机位姿变换
        cam_positions, focal_positions, cam_names = [], [], []
        for cam_type, cam_info in info['cams'].items():
            cam_names.append(cam_type)
            cam2ego = rt2mat(cam_info['sensor2ego_translation'], cam_info['sensor2ego_rotation'])
            f = 0.0055
            cam_position = cam2ego @ np.array([0., 0., 0., 1.]).reshape([4, 1])  # 相机位置
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = cam2ego @ np.array([0., 0., f, 1.]).reshape([4, 1])  # 相机的焦点位置
            focal_positions.append(focal_position.flatten()[:3])

        # occ_canvas
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
        # voxel_show = pred_occ != FREE_LABEL
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0])

        if args.draw_gt:
            voxel_show = np.logical_and(voxel_label != FREE_LABEL, camera_mask)
            vis = show_occ(torch.from_numpy(voxel_label), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                           offset=[0, voxel_label.shape[0] * voxel_size[0] * 1.2 * 1, 0])

        out_dir = os.path.join(vis_dir, f'{scene_name}', f'{cnt:04d}_{sample_token}')
        overall_img = draw(vis, imgs, out_dir, args.format, 
                       cam_positions=cam_positions, focal_positions=focal_positions, cam_names=cam_names, 
                       canva_size=canva_size, scale_factor=scale_factor, mode='pred')

        vis.clear_geometries()
        
        if args.format == 'video':
            cv2.putText(overall_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(overall_img, f'{scene_name}', (5, 35), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(overall_img, f'{sample_token[:5]}', (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(overall_img)

    if args.format == 'video':
        vout.release()
    
    print('Finish visualizing results !')
    vis.destroy_window()


if __name__ == '__main__':
    setproctitle.setproctitle("dzp_vis")
    main()