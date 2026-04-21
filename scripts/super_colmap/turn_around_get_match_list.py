import numpy as np
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import os
from os import listdir
from os.path import join, isfile
import re
import open3d as o3d
#得到延外参方向距离相机distance的点的坐标
def calculate_point_from_camera_center(camera_pose_matrix, distance):
    # Extract camera position from camera pose matrix
    camera_position = camera_pose_matrix[:3, 3]

    # Extract camera rotation matrix from camera pose matrix
    camera_rotation_matrix = camera_pose_matrix[:3, :3]

    # Compute the direction vector along the vertical image plane
    vertical_direction = np.array([0, 0, 1])  # Assuming the camera is pointing upwards
    camera_to_point_direction = np.dot(camera_rotation_matrix, vertical_direction)

    # Normalize the direction vector
    camera_to_point_direction /= np.linalg.norm(camera_to_point_direction)

    # Calculate the point at the specified distance along the direction vector
    point = camera_position + (distance * camera_to_point_direction)

    return point

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
    #print( snapshot_id)
    return snapshot_id

#读取images.txt里面每一个图片对应的车身的位姿和他所对应的外参
#以3*n*4*4的array，分别存储n个车身位姿和n个外参和n个相机位姿
def get_image_snapshot_pose_extri(fold_path,turn_point_image_id):
    car_txt_path = fold_path + '/car_poses.txt'
    image_txt_path = fold_path + '/images.txt'
    index_txt_path = fold_path + '/index.txt'
    cam_extris = get_camera_extrinsic_matrix(fold_path)
    ref_image_car_poses = []
    ref_image_extris = []
    ref_image_poses = []
    merge_image_car_poses = []
    merge_image_extris = []
    merge_image_poses = []
    car_poses = {}
    timestamp = {}
    with open(index_txt_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        snapshot_id = line[0]
        for j in range(2, len(line)):
            timestamp[line[j]] = snapshot_id
    with open(car_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        name = line[-1]
        snapshot_id = get_image_snapshot_id(name)
        quaternion = np.array([ float(line[2]), float(line[3]), float(line[4]),float(line[1])])
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
    num_images = len(lines)
    num_snapshot = num_images/6
    print(num_snapshot)
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        name = line[-1]
        snapshot_id = timestamp[name]
        image_id = int(line[0])
        camera_id = int(line[-2])
        #qx,qy,qz,qw
        quaternion = np.array([ float(line[2]), float(line[3]), float(line[4]),float(line[1])])
        translation = np.array([float(line[5]), float(line[6]), float(line[7])])
        image_pose = np.eye(4)
        rqut = Rotation.from_quat(quaternion)
        rotation = rqut.as_matrix()
        image_pose[:3, :3] = rotation
        image_pose[:3, 3] = translation
        image_pose = np.linalg.inv(image_pose)

        #print('image_id',image_id,'yushu',image_id%num_snapshot)
        if image_id%num_snapshot>turn_point_image_id:
            ref_image_car_poses.append(car_poses[snapshot_id])
            ref_image_extris.append(cam_extris[camera_id - 1])
            ref_image_poses.append(image_pose)
        else:
            merge_image_car_poses.append(car_poses[snapshot_id])
            merge_image_extris.append(cam_extris[camera_id - 1])
            merge_image_poses.append(image_pose)
    ref_poses_array = np.stack((ref_image_car_poses, ref_image_extris,ref_image_poses), axis=0)
    merge_poses_array = np.stack((merge_image_car_poses, merge_image_extris,merge_image_poses), axis=0)
    #print("ref_poses_array", ref_poses_array.shape)
    #print("merge_poses_array", merge_poses_array.shape)
    return ref_poses_array, merge_poses_array


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


def get_intrinsic_matrix(fold_path):
    images_txt_path = fold_path + '/images.txt'
    cameras_txt_path = fold_path + '/cameras.txt'
    camera_intrinsic_matrix = []
    image_intrinsic_matrix = []
    with open(cameras_txt_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        fx, fy, cx, cy = float(line[4]), float(line[5]), float(line[6]), float(line[7])
        camera_intrinsic_matrix.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        camera_id = int(line[-2])
        image_intrinsic_matrix.append(camera_intrinsic_matrix[camera_id - 1])
    image_intrinsic_matrix = np.stack(image_intrinsic_matrix, axis=0)
    # print("image_intrinsic_matrix", image_intrinsic_matrix.shape)
    return image_intrinsic_matrix


def compute_camera_frustum(image_shape, intrinsic_matrix, extrinsic_matrix):
    (W, H) = image_shape
    pose = extrinsic_matrix
    # 取平移向量
    C = pose[:3, 3].reshape([3, 1])
    # print("c", C.shape, C)
    # uv表示每一个像素点的坐标，3行，列数与像素点个数相同
    # 取四个角点
    uv = np.array([[0, 0, W, W], [0, H, 0, H], [1, 1, 1, 1]])

    K_inv = np.linalg.inv(intrinsic_matrix)
    # K_inv@Puv = [R T]@Pw
    ray = K_inv @ uv
    # 变换为4*像素点个数的矩阵
    ray = np.vstack([ray, np.ones_like(ray[:1, :])])
    # 4*4矩阵@4*像素点个数矩阵 = 4*像素点个数矩阵
    ray = pose @ ray
    # 三维点坐标减去相机的坐标
    ray = ray[:3, :] - C
    # print("ray", ray.shape, ray)
    point1 = C + 75 * ray
    # print("point1", point1.shape, point1)
    return C, point1

#给每一个相机寻找与其最近（比较的是车身位姿）的24个图片
#得到n*24的array，记录每一张图片最近的24个图片
def get_neighbours(poses_matrix,k):
    points = poses_matrix[0][:, :, 3][:, :3]

    tree = cKDTree(points)
    closest = tree.query(points, k=k)[1]
    # print("closest", closest.shape, closest)
    # 创建一个索引矩阵，每行都是0到k-1的整数
    indices = np.tile(np.arange(poses_matrix.shape[1]).reshape(-1, 1), (1, k))
    # print("indices", indices.shape, indices)
    # 使用索引矩阵去掉对应自身点的索引
    closest_without_self = np.where(closest == indices, -1, closest)
    #print("closest_without_self", closest_without_self.shape, closest_without_self)
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

#对于距离相机前48近的图片，计算它们与相机之间的角度差异
#以一个n*48*2的array记录，第一个数据是测试相机测image_id
#第二个数据是测试相机与参考相机的旋转角度差异
def rotation_angle_difference(poses_matrix,k):
    closest = get_neighbours(poses_matrix,k)
    rot_matrix = poses_matrix[0][:, :3, :3]
    extris_rot_matrix = poses_matrix[1][:, :3, :3]
    result = np.zeros((closest.shape[0], closest.shape[1]), dtype=np.float32)
    for i in range(closest.shape[0]):  # 对于参考相机i
        for j in range(closest.shape[1]):  # 对于距离参考相机i第j近的测试相机
            pose1 = rot_matrix[i] @ extris_rot_matrix[i]  # 时刻i的位姿
            if (closest[i, j] == -1):#-1代表是自己本身
                result[i][j] = 10
                continue
            pose2 = rot_matrix[closest[i, j]] @ extris_rot_matrix[closest[i, j]]  # 测试相机的位姿
            angle_diff = rotation_two_image_angle_difference(pose1, pose2)
            result[i][j] = angle_diff
    angel_diff = np.stack((closest, result), axis=2)
    #print("angel_diff", angel_diff.shape, angel_diff)
    return angel_diff

#输出一个n*k的数组，记录对于图像i，它所需要匹配的图像id
def get_match(angel_diff):
    # print(angel_diff.shape, angel_diff)
    match_list = [[] for _ in range(angel_diff.shape[0])]
    for i in range(angel_diff.shape[0]):
        for j in range(angel_diff.shape[1]):
            if angel_diff[i][j][1] < 1 and angel_diff[i][j][0] > i:#如果角度较小，且测试相机的id大于参考相机
                match_list[i].append(int(angel_diff[i][j][0]))
    #print(match_list)
    return match_list

'''def get_points(fold_path, ref_match_list):
    ref_poses, merge_poses = get_image_snapshot_pose_extri(fold_path)
    ref_points = ref_poses[0][:, :, 3][:, :3]
    merge_points = merge_poses[0][:, :, 3][:, :3]
    ref_num = len(ref_points)
    print(ref_num)
    num_nearest_points = 30  # 最近的点的数量
    nearest_points_indices = []

    merge_kdtree = cKDTree(merge_points)  # 构建merge_points的KD树

    for ref_point in ref_points:
        _, indices = merge_kdtree.query(ref_point, k=num_nearest_points)  # 查询最近的点及其索引
        nearest_points_indices.append(indices)

    for i, indices in enumerate(nearest_points_indices):
        #print("image_id:",i+1)
        for j in range(len(indices)):
            ref_match_list[i].append(int(indices[j]+ref_num))
            #print(indices[j])
            #print("match_image_id:",indices[j]+ref_num)
    #print(ref_match_list)
    return ref_match_list'''

def get_points(fold_path, ref_match_list,turn_point_image_id):
    ref_poses, merge_poses = get_image_snapshot_pose_extri(fold_path,turn_point_image_id)
    ref_num = len(ref_poses[0])
    ref_image_poses = ref_poses[2]
    merge_image_poses = merge_poses[2]
    ref_points= []
    merge_images = []

    print(ref_num)
    #存入点云
    for i in range(len(ref_image_poses)):
        ref_points.append(calculate_point_from_camera_center(ref_image_poses[i],20))
    for j in range(len(merge_image_poses)):
        merge_images.append(calculate_point_from_camera_center(merge_image_poses[j],20))
    for m in range(len(ref_image_poses)):
        count = 0
        #print("image_id:",m+1)
        for n in range(len(merge_images)):
            if  np.linalg.norm(ref_points[m]-merge_images[n])<10:
                if count>=12:
                    continue
                #print("match_image_idd:",n+ref_num)
                ref_match_list[m].append(n+ref_num)
                count+=1

    return ref_match_list




def turn_around_get_match_list(fold_path):
    #得到内部的匹配对
    with open(join(fold_path,'turn_point_id.txt'),'r') as f:
        turn_point_image_id= int(f.readline().strip())
    ref_poses,merge_poses = get_image_snapshot_pose_extri(fold_path,turn_point_image_id)
    ref_merged_array = rotation_angle_difference(ref_poses,24)
    ref_match_list = get_match(ref_merged_array)
    merge_merged_array = rotation_angle_difference(merge_poses,36)
    num_ref = len(ref_poses[0])
    num_merge = len(merge_poses[0])
    merge_match_list = get_match(merge_merged_array)
    match_list = []
    get_points(fold_path,ref_match_list,turn_point_image_id)
    merge_match_list = [[i+num_ref for i in merge_match_list[j]] for j in range(len(merge_match_list))]
    for i in range(num_ref):
        match_list.append(ref_match_list[i])
    for i in range(num_merge):
        match_list.append(merge_match_list[i])
    for i in range(len(match_list)):
        print("image_id:",i)
        print(match_list[i])
    #print(match_list[497])
    #print(match_list[498])
    return match_list
    #把merge_match_list加在ref_match_list后面






if __name__ == '__main__':
    fold_path = '/dataset/sfm/dataset/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025107_39.mcap/sfm/chouzhen'
    turn_point_image_id = 136
    turn_around_get_match_list(fold_path)
    #ref_poses,merge_poses = get_image_snapshot_pose_extri(fold_path)

