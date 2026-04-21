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
    #print( snapshot_id)
    return snapshot_id

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

#读取images.txt里面每一个图片对应的车身的位姿和他所对应的外参
#以2*n*4*4的array，分别存储n个车身位姿和n个外参
def get_image_snapshot_pose_extri(fold_path):
    car_txt_path = fold_path + '/car_poses.txt'
    image_txt_path = fold_path + '/images.txt'
    index_txt_path = fold_path + '/index.txt'
    cam_extris = get_camera_extrinsic_matrix(fold_path)
    image_car_poses = []
    image_extris = []
    car_poses = {}
    timestamp = {}
    snapshot_id_list = []
    snp_imgs = {}
    imgs_id_camid = {}
    with open(index_txt_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        snapshot_id = line[0]
        snp_imgs[snapshot_id] = []
        for j in range(2, len(line)):
            timestamp[line[j]] = snapshot_id
            snp_imgs[snapshot_id].append(line[j])
    # print(timestamp)
    # print(snp_imgs['69'])
    # exit()
    with open(car_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        name = line[-1]
        snapshot_id = get_image_snapshot_id(name)
        snapshot_id_list.append(snapshot_id)
        #qx,qy,qz,qw
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
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        name = line[-1]
        snapshot_id = timestamp[name]
        img_id = int(line[0]) - 1
        camera_id = int(line[-2])
        imgs_id_camid[name] = (img_id, camera_id)
        image_car_poses.append(car_poses[snapshot_id])
        image_extris.append(cam_extris[camera_id - 1])
    poses_array = np.stack((image_car_poses, image_extris), axis=0)
    # print('imgs_cam_id########################', imgs_cam_id)
    # exit()


    return poses_array, timestamp, snapshot_id_list, snp_imgs, imgs_id_camid

def process_per_img(fold_path, timestamp, snapshot_id_list, snp_imgs, imgs_id_camid, poses_matrix):
    image_txt_path = fold_path + '/images.txt'
    pair_list = {}
    rot_matrix = poses_matrix[0][:, :3, :3]
    extris_rot_matrix = poses_matrix[1][:, :3, :3]
    with open(image_txt_path, 'r') as f:
        lines = f.readlines()[::2]
    for i, line in enumerate(lines):
        pair_list_single = []
        diff_list = []
        line = line.strip().split(' ')
        name = line[-1]
        snp_id_0 = int(timestamp[name])
        camera_id_0 = int(line[-2]) ##当前图像视角
        img_id_0 = int(line[0]) - 1 ##当前图像id
        pose0 = rot_matrix[i] @ extris_rot_matrix[i]  # 当前图像位姿

        pair_list[img_id_0] = []
        ####获取距离当前时刻最近的14（同视角视角）个相机组snp_id，与最近的3个（相邻视角）
        for snp_id in snapshot_id_list:
            snp_id_1 = int(snp_id)
            diff = abs(snp_id_1 - snp_id_0)
            diff_list.append((snp_id_1, diff))
        diff_list.sort(key=lambda x: x[1])
        group_same_view = [str(snp_id_1) for snp_id_1, diff in diff_list[:15] if str(snp_id_1) != str(snp_id_0)]
        group_other_view = [str(snp_id_1) for snp_id_1, diff in diff_list[:3]]
        ####处理同视角图片
        for snp_id_1 in group_same_view:
            imgs = snp_imgs[snp_id_1]  # 参考时刻的图像
            for img in imgs:
                if img not in imgs_id_camid:  # 检查img是否在imgs_cam_id字典中
                    continue
                camera_id_1 = imgs_id_camid[img][1]  ##参考图像视角
                if camera_id_1 == camera_id_0:
                    img_id_1 = imgs_id_camid[img][0] ##参考图像id
                    pair_list_single.append((img_id_0, img_id_1))

        ####处理相邻视角图片
        for snp_id_1 in group_other_view:
            # print('snp_id_1::::', snp_id_1)
            imgs = snp_imgs[snp_id_1]
            # print('imgs::::', imgs)
            for img in imgs:
                if img not in imgs_id_camid:
                    continue
                camera_id_1 = imgs_id_camid[img][1]
                if camera_id_1 != camera_id_0:
                    img_id_1 = imgs_id_camid[img][0]  ##参考图像id
                    pose1 = rot_matrix[img_id_1] @ extris_rot_matrix[img_id_1]  # 参考图像位姿
                    angle_diff = rotation_two_image_angle_difference(pose0, pose1)
                    if angle_diff < 1:
                        pair_list_single.append((img_id_0, img_id_1))
    #     print('pair_list_single', pair_list_single)
    #     print('pair_list_single', len(pair_list_single))
    # exit()
        pair_list[img_id_0] = pair_list_single
    # print('pair_list', pair_list)
    return pair_list

def calculate_score(pair_list, score):
    for key in pair_list.keys():
        pair = pair_list[key]
        s = 500 ###任意赋值不小于0的分数
        for i in range(len(pair)):
            score[pair[i][0]][pair[i][1]] = s
            s = s-5

    return score

def get_score(fold_path, score):
    poses_array, timestamp, snapshot_id_list, snp_imgs, imgs_id_camid = get_image_snapshot_pose_extri(fold_path)
    pair_list = process_per_img(fold_path, timestamp, snapshot_id_list, snp_imgs, imgs_id_camid, poses_array)
    score = calculate_score(pair_list, score)

    return score

if __name__ == '__main__':
    fold_path = '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm_pair_test/chouzhen'
    num_images = 882
    score = np.zeros((num_images, num_images))
    get_score(fold_path, score)
