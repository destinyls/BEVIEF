import os
import cv2
import time

import mmcv
import math
import numpy as np
from pypcd import pypcd
from tqdm import tqdm

from scripts.vis_utils import *

def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points

def to_gt_boxes(ann_infos):
    gt_boxes = []
    for anno in ann_infos:
        x, y ,z = anno["translation"]
        yaw_lidar = anno["yaw_lidar"]
        l, w, h = anno["size"]
        gt_boxes.append([x, y, z, l, w, h, yaw_lidar])    
    gt_boxes = np.array(gt_boxes)
    return gt_boxes

def get_cam2virtual(denorm):
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    rot_mat, _ = cv2.Rodrigues(n_vector * sita)
    rot_mat = rot_mat.astype(np.float32)
    cam2virtual = np.eye(4)
    cam2virtual[:3, :3] = rot_mat
    return cam2virtual

if __name__ == '__main__':
    data_root = 'data/dair-v2x'
    info_path = 'data/dair-v2x/dair_12hz_infos_train.pkl'
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt'))
    infos = mmcv.load(info_path)
    for info in tqdm(infos):
        sample_id = info["sample_token"].split('/')[1].split('.')[0]
        lidar_info = info["lidar_infos"]["LIDAR_TOP"]
        lidar_path = lidar_info["filename"]

        lidar_file_path = os.path.join(data_root, lidar_path)        
        camera_info = info["cam_infos"]["CAM_FRONT"]
        calibrated_sensor = camera_info["calibrated_sensor"]
        camera_intrinsic = calibrated_sensor["camera_intrinsic"]
        rotation_matrix = calibrated_sensor["rotation_matrix"]
        translation = calibrated_sensor["translation"]
        denorm = camera_info["denorm"]
        cam2virtual = get_cam2virtual(denorm)
        height_ref = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
        
        cam2lidar = np.eye(4)
        cam2lidar[:3,:3] = rotation_matrix
        cam2lidar[:3, 3] = translation.flatten()
        lidar2cam = np.linalg.inv(cam2lidar)
        r_lidar2cam = lidar2cam[:3, :3]
        t_lidar2cam = lidar2cam[:3, 3]
        
        points = read_pcd(lidar_file_path)
        points[:, 3] = 1.0
        camera_points = np.matmul(lidar2cam, points.T).T
        virtual_points = np.matmul(cam2virtual, camera_points.T).T
        virtual_height = virtual_points[:, 1]
        # height_offsets = virtual_points[:, 1] - height_ref
        
        depths = camera_points[:, 2]
        P = np.eye(4)
        P[:3, :3] = camera_intrinsic        
        img_poins = np.matmul(P, camera_points.T)
        img_poins = img_poins[:2, :] / img_poins[2, :]
        
        img_shape, min_dist = (1080, 1920), 3
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, img_poins[0, :] > 1)
        mask = np.logical_and(mask, img_poins[0, :] < img_shape[1] - 1)
        mask = np.logical_and(mask, img_poins[1, :] > 1)
        mask = np.logical_and(mask, img_poins[1, :] < img_shape[0] - 1)
        img_poins = img_poins[:, mask].astype(np.int32)
        depths = depths[mask]
        virtual_height = virtual_height[mask]
        img_path = os.path.join(data_root, info["sample_token"])
        # img = cv2.imread(img_path)
        # img[img_poins[1,:], img_poins[0,:], 0] = 255
        # cv2.imwrite("demo.jpg", img)

        gt_boxes = to_gt_boxes(info["ann_infos"])
        # demo(img_path, gt_boxes, r_lidar2cam, t_lidar2cam, camera_intrinsic)
        os.makedirs(os.path.join(data_root, 'depth_gt'), exist_ok=True)
        os.makedirs(os.path.join(data_root, 'height_gt'), exist_ok=True)
        
        np.concatenate([img_poins[:2, :].T, depths[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(data_root, 'depth_gt',
                                        f'{sample_id}.jpg.bin'))
                       
        np.concatenate([img_poins[:2, :].T, virtual_height[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(data_root, 'height_gt',
                                        f'{sample_id}.jpg.bin'))
