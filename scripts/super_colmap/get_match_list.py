import numpy as np
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import os
from os import listdir
from os.path import join, isfile
import re


def get_image_snapshot_id(name):
    # print(name)
    removed_name = re.sub(r'\.jpg$', '', name)
    # 如果还有字母
    if re.search('[a-zA-Z]', removed_name):
        first_letter_index = re.search('[a-zA-Z]', removed_name).start()
        last_letter_index = re.search('[a-zA-Z]', removed_name[::-1]).start()
        last_letter_index = len(removed_name) - last_letter_index - 1
        useless_str = removed_name[first_letter_index:last_letter_index + 2]
        snapshot_id = re.sub(useless_str, '', removed_name)
    else:
        snapshot_id = removed_name
    # print( snapshot_id)
    return snapshot_id


# 读取images.txt里面每一个图片对应的车身的位姿和他所对应的外参
# 以2*n*4*4的array，分别存储n个车身位姿和n个外参
def get_image_snapshot_pose_extri(fold_path):
    car_txt_path = fold_path + '/car_poses.txt'
    image_txt_path = fold_path + '/images.txt'
    index_txt_path = fold_path + '/index.txt'
    cam_extris = get_camera_extrinsic_matrix(fold_path)
    image_car_poses = []
    image_extris = []
    car_poses = {}
    timestamp = {}
    with open(index_txt_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        snapshot_id = line[0]
        for j in range(2, len(line)):
            timestamp[line[j]] = snapshot_id
    #print(timestamp)
    with open(car_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        name = line[-1]
        snapshot_id = get_image_snapshot_id(name)
        #qx,qy,qz,qw
        quaternion = np.array([float(line[2]), float(line[3]), float(line[4]), float(line[1])])
        translation = np.array([float(line[5]), float(line[6]), float(line[7])])
        pose = np.eye(4)
        rqut = Rotation.from_quat(quaternion)
        rotation = rqut.as_matrix()
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        car_poses[snapshot_id] = pose
    # print(car_poses)
    with open(image_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        name = line[-1]
        #print(name)
        snapshot_id = timestamp[name]
        camera_id = int(line[-2])
        image_car_poses.append(car_poses[snapshot_id])
        image_extris.append(cam_extris[camera_id - 1])
    poses_array = np.stack((image_car_poses, image_extris), axis=0)
    #print("poses_array", poses_array)
    return poses_array


def get_camera_extrinsic_matrix(fold_path):
    cam_pose_txt_path = join(fold_path, 'ref_cam_pose.txt')
    cam_extris = []
    with open(cam_pose_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        quaternion = np.array([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
        translation = np.array([float(line[5]), float(line[6]), float(line[7])])
        pose = np.eye(4)
        rqut = Rotation.from_quat(quaternion)
        rotation = rqut.as_matrix()
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        pose = np.linalg.inv(pose)
        cam_extris.append(pose)
    return cam_extris


def get_pose_matrix(fold_path):
    images_txt_path = fold_path + '/images.txt'
    poses = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        quaternion = np.array([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
        translation = np.array([float(line[5]), float(line[6]), float(line[7])])
        pose = np.eye(4)
        rqut = Rotation.from_quat(quaternion)
        rotation = rqut.as_matrix()
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        pose = np.linalg.inv(pose)
        poses.append(pose)
    poses_array = np.stack(poses, axis=0)
    # print("poses", poses_array.shape)
    return poses_array


# 给每一个相机寻找与其最近（比较的是车身位姿）的48个图片
# 得到n*48的array，记录每一张图片最近的48个图片
def get_neighbours(poses_matrix, max_distance=10):
    points = poses_matrix[0][:, :, 3][:, :3]
    # print(poses_matrix.shape, points.shape)
    tree = cKDTree(points)
    k = min(24, points.shape[0] - 5)
    distances, closest = tree.query(points, k=k)

    # 筛选距离不大于max_distance的邻居点
    mask = distances <= max_distance
    closest_in_range = closest.copy()
    # 距离不在范围内的点设置为-1
    closest_in_range[~mask] = -1
    # 创建一个索引矩阵，每行都是0到k-1的整数
    indices = np.tile(np.arange(poses_matrix.shape[1]).reshape(-1, 1), (1, k))
    # print("indices", indices.shape, indices)
    # 使用索引矩阵去掉对应自身点的索引
    closest_without_self = np.where(closest_in_range == indices, -1, closest_in_range)
    # print("closest_without_self", closest_without_self.shape, closest_without_self)
    return closest_without_self


def rotation_two_image_angle_difference(rot_matrix1, rot_matrix2):
    # 计算旋转矩阵的转置，以便使用np.dot计算两个矩阵的乘积
    rot1_transpose = np.transpose(rot_matrix1)
    # 计算两个旋转矩阵的乘积
    rot_matrix_product = np.dot(rot1_transpose, rot_matrix2)
    # 计算旋转矩阵的迹
    trace = np.trace(rot_matrix_product)
    # 限制迹的取值范围在[-1, 3]之间，避免浮点数运算误差引起的错误
    trace = np.clip(trace, -1, 3)
    # 计算旋转角度差异
    angle_difference = np.arccos((trace - 1) / 2)

    return angle_difference


# 对于距离相机前48近的图片，计算它们与相机之间的角度差异
# 以一个n*48*2的array记录，第一个数据是测试相机测image_id
# 第二个数据是测试相机与参考相机的旋转角度差异
def rotation_angle_difference(poses_matrix):
    closest = get_neighbours(poses_matrix)
    rot_matrix = poses_matrix[0][:, :3, :3]
    extris_rot_matrix = poses_matrix[1][:, :3, :3]
    result = np.zeros((closest.shape[0], closest.shape[1]), dtype=np.float32)
    for i in range(closest.shape[0]):  # 对于参考相机i
        for j in range(closest.shape[1]):  # 对于距离参考相机i第j近的测试相机
            pose1 = rot_matrix[i] @ extris_rot_matrix[i]  # 时刻i的位姿
            if closest[i, j] == -1:  # -1代表是自己本身或者距离过远
                result[i][j] = 10
                continue
            pose2 = rot_matrix[closest[i, j]] @ extris_rot_matrix[closest[i, j]]  # 测试相机的位姿
            angle_diff = rotation_two_image_angle_difference(pose1, pose2)
            result[i][j] = angle_diff
    angel_diff = np.stack((closest, result), axis=2)
    # print("angel_diff", angel_diff.shape, angel_diff)
    return angel_diff


# 输出一个n*k的数组，记录对于图像i，它所需要匹配的图像id
def get_match(angel_diff):
    # print(angel_diff.shape, angel_diff)
    match_list = [[] for _ in range(angel_diff.shape[0])]
    for i in range(angel_diff.shape[0]):
        for j in range(angel_diff.shape[1]):
            # 如果角度较小，且测试相机的id大于参考相机，说明是需要匹配的
            if angel_diff[i][j][1] < 1 and angel_diff[i][j][0] > i:
                match_list[i].append(int(angel_diff[i][j][0]))
    # print(match_list)
    return match_list


def get_match_list(fold_path):
    image = get_image_snapshot_pose_extri(fold_path)
    merged_array = rotation_angle_difference(image)
    match_list = get_match(merged_array)
    return match_list


if __name__ == '__main__':
    fold_path = '/vepfs_dataset/sjtu/J3/fco53urrc77u172f2elbg-EP32_J3_9YQ_drive_merge-trigger-20240401-125940-2098.tar.encrypted/sfm'
    #fold_path = "/vepfs_dataset/sjtu/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-02-04-43-48_81/sfm"
    get_match_list(fold_path)
