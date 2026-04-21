import os
import cv2
import numpy as np
import open3d as o3d
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from os.path import join, isfile, isdir
from scipy.spatial.transform import Rotation
import argparse
import re
import math
import shutil
from scipy.spatial import KDTree


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


def get_ref_poses(path):
    car_poses = {}
    snapshot_id_list = []
    if not isfile(path):
        return car_poses

    with open(path, 'r') as f:
        lines = f.readlines()[::2]

    for line in lines:
        line = line.strip().split(' ')
        name = line[-1]
        snapshot_id = get_image_snapshot_id(name)
        snapshot_id_list.append(snapshot_id)
        # qx,qy,qz,qw
        quaternion = np.array([float(line[2]), float(line[3]), float(line[4]), float(line[1])])
        translation = np.array([float(line[5]), float(line[6]), float(line[7])])
        pose = np.eye(4)
        rqut = Rotation.from_quat(quaternion)
        rotation = rqut.as_matrix()
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        car_poses[snapshot_id] = pose
    snapshot_id_list = sorted(snapshot_id_list, key=lambda x: int(x))

    return car_poses, snapshot_id_list
#######计算每个点的曲率
def calculate_curvature(scene, sfm_folder='sfm'):
    car_pose_path = join(scene, sfm_folder, 'chouzhen', 'car_poses.txt')
    poses, snapshot_id_list = get_ref_poses(car_pose_path)

    # print('snapshot_id_list', snapshot_id_list)
    curvature_snp = []
    turn_list = []
    for key in poses.keys():
        num_key = len(snapshot_id_list)
        index_0 = snapshot_id_list.index(key)
        if index_0 > num_key - 3:
            continue
        index_1 = index_0 + 1
        index_2 = index_0 + 2

        key1 = snapshot_id_list[index_1]
        key2 = snapshot_id_list[index_2]

        pose0 = poses[key][:3, 3]
        pose1 = poses[key1][:3, 3]
        pose2 = poses[key2][:3, 3]

        # 将点的坐标转换为numpy数组
        point0 = np.array(pose0[:2])
        point1 = np.array(pose1[:2])
        point2 = np.array(pose2[:2])

        # 计算两个向量之间的夹角
        angle = np.arccos(np.dot(point1 - point0, point2 - point0) / (
                np.linalg.norm(point1 - point0) * np.linalg.norm(point2 - point0)))

        # 计算曲率
        curvature = 2 * np.sin(angle) / np.linalg.norm(point2 - point0)
        curvature_snp.append((key, curvature))
    filtered_curvature_snp = [(key, c) for key, c in curvature_snp if c > 0.05]
    return filtered_curvature_snp, snapshot_id_list

#######判断曲率是否符合要求，去除离散的点
def judgement(filtered_curvature_snp, snapshot_id_list):
    sorted_filtered_curvature_snp = sorted(filtered_curvature_snp, key=lambda x: x[1], reverse=True)
    turn_list = [item[0] for item in sorted_filtered_curvature_snp]
    turn_id_list = [snapshot_id_list.index(turn_key) for turn_key in turn_list]

    # 构建KD树,去除离散的点
    kdtree = KDTree(np.array(turn_id_list).reshape(-1, 1))
    # 查询距离大于阈值的点作为离散点
    threshold = 5
    discrete_points = set()
    for i in range(len(turn_id_list)):
        idx = turn_id_list[i]
        neighbors = kdtree.query_ball_point(idx, threshold)
        if len(neighbors) <= 3:  # 自身也算一个邻居
            discrete_points.add(idx)
    # 去除离散点
    filtered_turn_id_list = [idx for idx in turn_id_list if idx not in discrete_points]
    filtered_turn_list = [snapshot_id_list[idx] for idx in filtered_turn_id_list]

    return filtered_turn_list


def get_ply(scene, sfm_folder='sfm', turn_list=None):

    car_pose_path = join(scene, sfm_folder, 'chouzhen', 'car_poses.txt')
    raw_ply = join(scene, sfm_folder, 'chouzhen', 'raw.ply')
    turn_ply = join(scene, sfm_folder, 'chouzhen', 'turn.ply')

    point_cloud_raw = o3d.geometry.PointCloud()
    point_cloud_raw.clear()
    point_cloud_turn = o3d.geometry.PointCloud()
    point_cloud_turn.clear()

    poses, _ = get_ref_poses(car_pose_path)

    print('save to ', raw_ply)

    for key in poses.keys():
        translation_vector = poses[key][:3, 3]
        points = np.zeros((1, 3))
        points[0] = translation_vector
        point_cloud_raw.points.extend(o3d.utility.Vector3dVector(points))
        if key in turn_list:
            point_cloud_turn.points.extend(o3d.utility.Vector3dVector(points))

    o3d.io.write_point_cloud(raw_ply, point_cloud_raw)
    o3d.io.write_point_cloud(turn_ply, point_cloud_turn)
    print('end')

####### 去掉掉头帧，将剩余的帧分割
def seg_turnaround(scene, sfm_folder='sfm', turn_list=None):
    car_pose_path = join(scene, sfm_folder, 'chouzhen', 'car_poses.txt')
    index_txt_path = join(scene, sfm_folder, 'chouzhen', 'index.txt')

    _, snapshot_id_list = get_ref_poses(car_pose_path)
    sorted_turn_list = sorted(turn_list, key=int)  # 按数字大小排序

    bound_index0 = snapshot_id_list.index(sorted_turn_list[0])
    bound_index1 = snapshot_id_list.index(sorted_turn_list[-1])
    snp_list0 = snapshot_id_list[:bound_index0]
    snp_list1 = snapshot_id_list[bound_index1 + 1:]

    snp_lists = [snp_list1, snp_list0]
    snp_lists = [snp_list for snp_list in snp_lists if len(snp_list) > 50]

    return snp_lists

####### 判断是否会生成两个clip，若为一个，则在本文件夹下修改文件；若分为两个clip，则i为0时，生成turnaround_scene文件夹并复制相关文件，软链接对应文件夹,当i为1则在本文件夹下修改文件
def get_clips(scene, sfm_folder='sfm', snp_lists=None):
    global scene_2
    raw_file_path = join(scene, sfm_folder, 'chouzhen')

    raw_imgs_lsit_path = join(raw_file_path, 'images_list.txt')
    raw_imgs_txt_path = join(raw_file_path, 'images.txt')
    raw_car_poses_path = join(raw_file_path, 'car_poses.txt')
    raw_index_txt_path = join(raw_file_path, 'index.txt')
    raw_params = join(raw_file_path, 'params')
    raw_cam_pose = join(raw_file_path, 'cam_pose.txt')
    raw_cameras = join(raw_file_path, 'cameras.txt')
    raw_points3D = join(raw_file_path, 'points3D.txt')
    raw_ref_cam_pose = join(raw_file_path, 'ref_cam_pose.txt')
    raw_rig_json = join(raw_file_path, 'rig.json')

    # print('snp_lists', snp_lists[1])
    # exit()
    for i, snp_list in enumerate(snp_lists):
        if len(snp_lists) > 1:
            if i == 1:
                target_file_path = join(scene, sfm_folder, 'chouzhen')
                print("创建turnaround-bag")
                # continue
            else:
                scene_2 = os.path.join(os.path.dirname(scene), 'turnaround_' + os.path.basename(scene))
                target_file_path = join(scene_2, sfm_folder, 'chouzhen')
                if not os.path.exists(target_file_path):
                    os.makedirs(target_file_path)
        else:
            target_file_path = join(scene, sfm_folder, 'chouzhen')
            # exit()
        # 创建目录
        os.makedirs(target_file_path, exist_ok=True)
        target_imgs_lsit_path = join(target_file_path, 'images_list.txt')
        target_imgs_txt_path = join(target_file_path, 'images.txt')
        target_car_poses_path = join(target_file_path, 'car_poses.txt')
        target_index_txt_path = join(target_file_path, 'index.txt')
        targrt_params = join(target_file_path, 'params')
        targrt_cam_pose = join(target_file_path, 'cam_pose.txt')
        targrt_cameras = join(target_file_path, 'cameras.txt')
        targrt_points3D = join(target_file_path, 'points3D.txt')
        targrt_ref_cam_pose = join(target_file_path, 'ref_cam_pose.txt')
        targrt_rig_json = join(target_file_path, 'rig.json')

        # 读取index.txt文件获取需要的图片名称
        with open(raw_index_txt_path, 'r') as file:
            lines = file.readlines()
            all_imgs = []
            for line in lines:
                snp_id, _, *img_names = line.split()
                if snp_id in snp_list:
                    all_imgs.extend(img_names)

        # 遍历raw_imgs_lsit_path文件
        with open(raw_imgs_lsit_path, 'r') as file:
            lines = file.readlines()
            selected_imgs = []
            for line in lines:
                img_name = line.strip()
                if img_name in all_imgs:
                    selected_imgs.append(line)

        # 写入选中的行到target_imgs_lsit_path文件中
        with open(target_imgs_lsit_path, 'w') as file:
            file.writelines(selected_imgs)

        # 生成需要的imgs_txt
        with open(raw_imgs_txt_path, 'r') as file:
            lines = file.readlines()
            with open(target_imgs_txt_path, 'w') as target_file:
                img_id = 1
                for line in range(0, len(lines), 2):
                    _, *data, img_name = lines[line].split()
                    if img_name.strip() in all_imgs:
                        target_file.write(f"{img_id} {' '.join(data)} {img_name}\n")
                        target_file.write(lines[line + 1])
                        img_id += 1

        # 读取原始车辆姿态数据并写入目标文件
        with open(raw_car_poses_path, 'r') as f:
            lines = f.readlines()
            with open(target_car_poses_path, 'w') as target_file:
                img_id = 1  # 重置img_id为1
                for line in lines:
                    if not line.strip():  # Check if the line is empty
                        continue
                    data = line.split()
                    snapshot_id = get_image_snapshot_id(data[-1])
                    if snapshot_id in snp_list:
                        target_file.write(f"{img_id} {' '.join(data[1:])}\n")
                        img_id += 1
                        target_file.write('\n')

        if i == 0 and len(snp_lists) > 1:
            # 复制文件
            shutil.copy2(raw_index_txt_path, target_index_txt_path)
            shutil.copy2(raw_cam_pose, targrt_cam_pose)
            shutil.copy2(raw_cameras, targrt_cameras)
            shutil.copy2(raw_points3D, targrt_points3D)
            shutil.copy2(raw_ref_cam_pose, targrt_ref_cam_pose)
            shutil.copy2(raw_rig_json, targrt_rig_json)
            if not os.path.exists(targrt_params):
                shutil.copytree(raw_params, targrt_params)

            if os.path.exists(join(scene, 'image')):
                raw_img_floder = join(scene, 'image')
                targrt_img_floder = join(scene_2, 'image')
            elif os.path.exists(join(scene, 'rawData')):
                raw_img_floder = join(scene, 'rawData')
                targrt_img_floder = join(scene_2, 'rawData')
            else:
                raw_img_floder = join(scene, 'rawCamera')
                targrt_img_floder = join(scene_2, 'rawCamera')

            if os.path.exists(join(scene, 'seg')):
                raw_seg_floder = join(scene, 'seg')
                targrt_seg_floder = join(scene_2, 'seg')
            else:
                raw_seg_floder = join(scene, 'seg2')
                targrt_seg_floder = join(scene_2, 'seg2')

            raw_files_floder = join(scene, 'files')
            targrt_files_floder = join(scene_2, 'files')

            os.symlink(raw_img_floder, targrt_img_floder)
            os.symlink(raw_seg_floder, targrt_seg_floder)
            os.symlink(raw_files_floder, targrt_files_floder)

def make_ply(scene, sfm_folder='sfm'):
    filtered_curvature_snp, snapshot_id_list = calculate_curvature(scene, sfm_folder)
    if not filtered_curvature_snp:
        print('no_turnaround')
    else:
        turn_list = judgement(filtered_curvature_snp, snapshot_id_list)
        print('turn_list', turn_list)
        get_ply(scene, sfm_folder, turn_list)


def make_clips(scene, sfm_folder='sfm'):
    filtered_curvature_snp, snapshot_id_list = calculate_curvature(scene, sfm_folder)
    if not filtered_curvature_snp:
        print('没有掉头帧')
        # clip_same_chouzhen(scene, sfm_folder)
    else:
        turn_list = judgement(filtered_curvature_snp, snapshot_id_list)
        if len(turn_list) <= 5:
            print('turn_list', turn_list, len(turn_list))
            print('曲率大的点过少，不进行分割')
        else:
            snp_lists = seg_turnaround(scene, sfm_folder, turn_list)
            get_clips(scene, sfm_folder, snp_lists)
            print('done')


if __name__ == '__main__':
    # scene = '/dataset/sfm/dataset/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025107_39.mcap'
    # make_ply(scene, 'sfm')
    # make_clips(scene, 'sfm')
    # make_one_clip(scene, 'sfm')

    # scene = '/dataset/sfm/dataset_merge/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025007_38'
    # make_ply(scene, 'sfm_418')
    # make_clips(scene, 'sfm_418')

    # scene = '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag'
    # make_ply(scene, 'sfm_pair_test')
    # make_clips(scene, 'sfm_pair_test')

    scene = '/dataset/sfm/dataset/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025107_39.mcap'
    # make_ply(scene, 'sfm')
    make_clips(scene, 'sfm')

