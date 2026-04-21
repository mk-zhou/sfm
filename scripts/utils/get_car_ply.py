import os
import cv2
import numpy as np
import open3d as o3d
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from os.path import join, isfile, isdir
from scipy.spatial.transform import Rotation


def get_ref_poses(scene):
    paths = [join(scene, 'files', 'sfm_pose_INTER_aligned_time_TUM.txt'),
             join(scene, "files", 'transed_sfm_pose_INTER_aligned_time_TUM.txt'),]
    types = ['sfm_tum', 'transed_sfm']
    car_poses_all_type = {}

    for path, type in zip(paths, types):
        if not isfile(path):
            continue
        with open(path, 'r') as f:
            lines = f.readlines()[::2]

        car_poses = []  # 创建每个类型的新位姿列表

        for line in lines:
            parts = line.strip().split(' ')
            if 'TUM' in path:
                t = np.array(parts[1:4], dtype=np.float64)
                q = np.array(parts[4:8], dtype=np.float64)
            else:
                q = np.array(parts[1:5], dtype=np.float64)
                t = np.array(parts[5:8], dtype=np.float64)

            car_pose = np.eye(4)
            car_pose[:3, :3] = Rotation.from_quat(q).as_matrix()
            car_pose[:3, 3] = t
            car_poses.append(car_pose)

        car_poses_all_type[type] = car_poses  # 将位姿列表存储在字典中的相应类型键下

    return car_poses_all_type


def get_car_ply(scene):
    point_cloud = o3d.geometry.PointCloud()

    car_poses_all_type = get_ref_poses(scene)
    for type, poses in car_poses_all_type.items():
        point_cloud.clear()
        car_ply = join(scene, type + '_car.ply')
        print('save to ', car_ply)
        for pose in poses:
            # 提取位姿矩阵中的旋转矩阵和平移向量
            translation_vector = pose[:3, 3]
            # 生成点云坐标
            points = np.zeros((1, 3))  # 假设每个位姿只有一个点
            points[0] = translation_vector
            # 将点云坐标添加到点云对象中
            point_cloud.points.extend(o3d.utility.Vector3dVector(points))
        o3d.io.write_point_cloud(car_ply, point_cloud)


if __name__ == '__main__':
    scene = '/dataset/rtfbag/merge_datasets/557039854_190/EP_has_go/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-022113_16'
    get_car_ply(scene)
