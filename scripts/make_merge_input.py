import numpy as np
import cv2
import os
from os import listdir
from os.path import join, isfile, isdir
import yaml
import shutil
import math
import re
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import json
import open3d as o3d
from tqdm import tqdm
import sys
import datetime
from scipy.spatial import cKDTree
import itertools
from collections import Counter

sys.path.append(('./super_colmap'))
sys.path.append(('./utils'))
from super_colmap.super_colmap import superpoint_geomatch, superpoint_geomatch_clip
from step import rig_mapper, get_road_points_from_scene, get_txt_from_scene, get_dense_ply_from_scene, \
    get_dense_ply_from_scene_clip, rig_mapper_clip, dense_ACMP_colmap, merge_mapper
from get_tum import get_tum_files
import argparse
from get_video_from_scene import get_video


# 生成database.db,与单个scene不同
def get_db(scene):
    superpoint_geomatch(scene, 'sfm', 1)


# 检查是否还有场景没有连通
def check_array(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == 0:
                for m in range(array.shape[0]):
                    if array[i, m] != 0 and array[m, j] != 0 and array[i, m] != -1 and array[m, j] != -1:
                        array[i, j] = 1
                        break


def dfs(matrix, visited, vertex, category, categories):
    visited[vertex] = True
    categories[vertex] = category

    for i in range(len(matrix)):
        if matrix[vertex, i] != 0 and not visited[i]:
            dfs(matrix, visited, i, category, categories)


def find_categories(matrix):
    n = matrix.shape[0]
    visited = [False] * n
    categories = [-1] * n
    category = 0

    for i in range(n):
        if not visited[i]:
            dfs(matrix, visited, i, category, categories)
            category += 1

    return categories


def find_largest_category_index(categories):
    category_counts = {}
    for category in categories:
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    largest_category_index = max(category_counts, key=category_counts.get)
    return largest_category_index


def find_vertices_in_largest_category(categories, largest_category_index):
    vertices_in_largest_category = [i for i, category in enumerate(categories) if category == largest_category_index]
    return vertices_in_largest_category


def get_sparse_ply(scene):
    ply_path = join(scene, 'sfm', 'chouzhen', 'rig_mapper', 'sparse.ply')
    input_path = join(scene, 'sfm', 'chouzhen', 'rig_mapper', '0')
    print(scene)

    os.system(f'colmap model_converter --input_path {input_path} --output_path {ply_path} --output_type PLY')


# 从iamges.txt中读取数据，返回一个字典，
# key是snapshot_id，是int类型，value是一个字典，
# key是camera_type，是字符串类型，value是一个list，字符串类型
# 0-3: quaternion, 4-6: translation
def get_data_from_images_txt(images_txt_path):
    poses = {}
    with open(images_txt_path, 'r') as first:
        lines = first.readlines()
    if lines[0].startswith('#'):
        location_lines = lines[4::2]
    else:
        location_lines = lines[::2]
    for line in location_lines:
        data = line.strip().split()
        name = data[-1]
        first_letter_index = re.search('[a-zA-Z]', name).start()
        last_underscore_index = name.rfind("_")
        if last_underscore_index != -1:
            snapshot_id = name[last_underscore_index + 1:name.rfind(".")]
            snapshot_id = int(snapshot_id)
            camera_type = name[first_letter_index:last_underscore_index]
            # print(camera_type, snapshot_id, name)
        else:
            snapshot_id = None
            camera_type = None
        if snapshot_id not in poses:
            poses[snapshot_id] = {}
        poses[snapshot_id][camera_type] = data[1:8]
    return poses


def get_data_from_car_pose_txt(car_pose_txt_path):
    poses = {}
    with open(car_pose_txt_path, 'r') as first:
        lines = first.readlines()
    if lines[0].startswith('#'):
        location_lines = lines[4::2]
    else:
        location_lines = lines[::2]
    for line in location_lines:
        data = line.strip().split()
        name = data[-1]
        snapshot_id = int(name[:name.rfind(".")])
        poses[snapshot_id] = data[1:8]
    return poses


def get_data_from_index_txt(index_txt_path):
    image_ids = []
    with open(index_txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        image_ids.append(int(line.strip()))
    return image_ids


def check_file_existence(file_path):
    if not isfile(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在。")


def load_json_file(file_path):
    check_file_existence(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class MergeInput:
    def __init__(self, yaml_path):
        self.ref_extrinsics = {}
        self.merge_extrinsics = {}
        self.yaml_path = yaml_path
        self.merge_pose_count = None
        self.ref_pose_count = None
        self.merge_camera_params_path = None
        self.ref_camera_params_path = None
        self.sfm_dir = None
        self.output_scene = None
        self.merge_scene = None
        self.ref_scene = None
        self.merge_scene_id = None
        self.ref_scene_id = None
        self.success = False
        with open(yaml_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.width = configs["image_width"]
        self.height = configs["image_height"]
        self.base_dir = configs["base_dir"]
        self.img_data_name = configs["img_data_name"]
        self.type = configs["type"]
        self.data_type = configs["data_type"]
        if 'orin' in self.data_type:
            self.camera_names = configs["orin_camera_names"]
        else:
            self.camera_names = configs["maptr_camera_names"]
        self.camera_types = configs["camera_types"]

        if self.data_type == "orin_soc":
            self.image_folder_prefix = 'soc_encoded_camera_'
            self.image_folder_suffix = ''
        elif self.data_type == "orin":
            self.image_folder_prefix = 'camera-'
            self.image_folder_suffix = ''
        elif self.type == "maptr" or self.type == "slam" or self.type == "dr_pgo" or self.type == "dr":
            self.image_folder_prefix = 'camera-'
            self.image_folder_suffix = ''
        else:
            self.image_folder_prefix = 'camera-'
            self.image_folder_suffix = '-encoder'
        self.distance_step = configs["distance_step"]
        self.camera_count = len(self.camera_names)

        self.camera2chassis = np.asarray([
            [1, 0, 0, -1.81],
            [0, 1, 0, 0],
            [0, 0, 1, -0.043],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        sub_files = sorted(os.listdir(self.base_dir))
        self.scenes = [os.path.join(self.base_dir, sub_file)
                       for sub_file in sub_files
                       if (os.path.isdir(os.path.join(self.base_dir, sub_file))
                           and os.path.isdir(os.path.join(self.base_dir, sub_file, 'files'))
                           and os.path.exists(os.path.join(self.base_dir, sub_file, 'files',
                                                           'sfm_pose_INTER_aligned_time_INDEX.txt'))

                           )]
        self.scene_ids = [sub_file
                          for sub_file in sub_files
                          if (os.path.isdir(os.path.join(self.base_dir, sub_file))
                              and os.path.isdir(os.path.join(self.base_dir, sub_file, 'files'))
                              and os.path.exists(os.path.join(self.base_dir, sub_file, 'files',
                                                              'sfm_pose_INTER_aligned_time_INDEX.txt'))

                              )]
        print(self.scenes)
        '''for scene_id in self.scene_ids:
            os.symlink(join(self.base_dir, scene_id),join('/vepfs_dataset/sjtu/IRMV/sfm/dataset_merge/road_test',scene_id))'''
        _, _, vertices_in_largest_category = self.find_valid_scenes()
        self.scene_ids = [self.scene_ids[i] for i in vertices_in_largest_category]
        self.scenes = [self.scenes[i] for i in vertices_in_largest_category]
        self.todo_scenes, self.done_scenes, _ = self.find_valid_scenes()
        for i in range(len(self.scenes)):
            for j in range(len(self.scenes)):
                if self.todo_scenes[i, j] == self.todo_scenes[j, i]:
                    self.todo_scenes[j, i] = 0
                elif self.todo_scenes[j, i] > self.todo_scenes[i, j] > 0:
                    self.todo_scenes[j, i] = 0
                elif self.todo_scenes[i, j] > self.todo_scenes[j, i] > 0:
                    self.todo_scenes[i, j] = 0

        #print(self.todo_scenes)
        #print(self.done_scenes)

    def find_valid_scenes(self):
        num_scenes = len(self.scenes)
        todo_scenes = np.zeros((num_scenes, num_scenes))
        score_matrix = np.zeros((num_scenes, num_scenes))
        done_scenes = np.zeros((num_scenes, num_scenes))
        for i in range(num_scenes):
            scores = []
            for j in range(num_scenes):
                score = self.score_scenes(self.scenes[i], self.scenes[j])
                score_matrix[i, j] = score
                score_matrix[j, i] = score
                scores.append(score)
            print(scores)
            # 对 scores 进行排序并记录在 self.todo_scenes 中
            sorted_scores_indices = np.argsort(scores)[::-1]
            for n in range(num_scenes):
                todo_scenes[i, sorted_scores_indices[n]] = n + 1
        todo_scenes[score_matrix == 0] = 0
        done_scenes[score_matrix == 0] = -1
        print(todo_scenes)
        categories = find_categories(todo_scenes)
        largest_category_index = find_largest_category_index(categories)
        vertices_in_largest_category = find_vertices_in_largest_category(categories, largest_category_index)
        return todo_scenes, done_scenes, vertices_in_largest_category

    def make_yaml_path(self):
        found = False
        for m in range(1, int(np.max(self.todo_scenes)) + 1):
            indices = np.argwhere(self.todo_scenes == m)
            for n in range(len(indices)):

                i = min(indices[n][0], indices[n][1])
                j = max(indices[n][0], indices[n][1])
                if self.done_scenes[i, j] == 0:
                    print("i", i, "j", j)
                    with open(self.yaml_path, 'r') as file:
                        data = yaml.safe_load(file)
                    # 修改YAML文件中的值
                    data['ref_scene_id'] = self.scene_ids[i]
                    data["merge_scene_id"] = self.scene_ids[j]
                    self.ref_scene_id = self.scene_ids[i]
                    self.merge_scene_id = self.scene_ids[j]
                    # 保存修改后的YAML文件
                    with open('temp.yaml', 'w') as file:
                        yaml.dump(data, file)
                    found = True
                    break  # 跳出内层循环
            if found:
                break

        self.ref_scene = join(self.base_dir, self.ref_scene_id)
        self.merge_scene = join(self.base_dir, self.merge_scene_id)
        self.output_scene = join(self.base_dir, "ref_" + self.ref_scene_id + "_merge_" + self.merge_scene_id)
        self.sfm_dir = join(self.output_scene, "sfm")
        print('sfm_dir:', self.sfm_dir)
        # print(self.done_scenes)
        self.ref_camera_params_path = join(self.ref_scene, 'sfm', 'chouzhen', 'params')
        self.merge_camera_params_path = join(self.merge_scene, 'sfm', 'chouzhen', 'params')
        self.ref_pose_count = -1
        self.merge_pose_count = -1
        for type, camera in zip(self.camera_types, self.camera_names):
            # 加载车体到相机外参
            camera_extrinsic_file = join(self.ref_camera_params_path, 'top_center_lidar-to-' + type + '-extrinsic.json')
            camera_extrinsic_data = load_json_file(camera_extrinsic_file)
            self.ref_extrinsics[camera] = np.array(
                camera_extrinsic_data["top_center_lidar-to-" + type + '-extrinsic']['param']['sensor_calib']['data'])

        for type, camera in zip(self.camera_types, self.camera_names):
            # 加载车体到相机外参
            camera_extrinsic_file = join(self.merge_camera_params_path,
                                         'top_center_lidar-to-' + type + '-extrinsic.json')
            camera_extrinsic_data = load_json_file(camera_extrinsic_file)
            self.merge_extrinsics[camera] = np.array(
                camera_extrinsic_data["top_center_lidar-to-" + type + '-extrinsic']['param']['sensor_calib']['data'])

        if not os.path.exists(self.output_scene):
            os.mkdir(self.output_scene)
        if not os.path.exists(self.sfm_dir):
            os.mkdir(self.sfm_dir)
        return i, j

    # 将sfm_pose_INTER_aligned_time_TUM.txt文件中的数据转换成车身位姿
    def loadarray_hozon(self, array):
        input_pose = array
        # print(input_pose)
        transforms = []
        timestamp = []
        for i, pose in enumerate(input_pose):
            transform = np.eye(4)
            translation = pose[1:4]
            rqut = Rotation.from_quat(pose[4:8])
            rotation = rqut.as_matrix()
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            # 得到车身位姿
            transform = transform @ self.camera2chassis
            timestamp.append(pose[0])
            '''if i == 0:
                print(pose[4:8])
                print(rqut)
                print(transform)'''
            transforms.append(transform)
        transforms = np.array(transforms)
        return transforms, timestamp

    # 读取sfm_pose_INTER_aligned_time_TUM.txt文件，得到车身位姿
    # 输入scene的地址
    # 返回车身位姿，np.array类型，n*4*4
    def get_scene_pose(self, scene):
        scene_dir = join(self.base_dir, scene)
        TUM_file_dir = join(scene_dir, 'files', 'sfm_pose_INTER_aligned_time_TUM.txt')
        car_poses, _ = self.loadarray_hozon(np.loadtxt(TUM_file_dir))
        return car_poses

    # 读取sfm_pose_INTER_aligned_time_INDEX.txt文件，得到图片的索引
    # 输入scene的地址
    # 返回图片的索引，list格式
    def get_scene_indexs(self, scene):
        scene_dir = join(self.base_dir, scene)
        index_file_path = join(scene_dir, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt')
        with open(index_file_path, 'r') as file:
            image_index = [line.strip() for line in file.readlines()]
        return image_index

    # 得到两个场景中sfm_pose_INTER_aligned_time_TUM.txt中前20接近的
    # 数据的平均距离
    # 返回平均距离
    def get_average_distance(self, scene_id1, scene_id2):
        scene1 = join(self.base_dir, scene_id1)
        scene2 = join(self.base_dir, scene_id2)
        ref_scene_pose = self.get_scene_pose(scene1)
        merge_scene_pose = self.get_scene_pose(scene2)
        ref_positions = ref_scene_pose[:, :3, 3]
        merge_positions = merge_scene_pose[:, :3, 3]
        ref_tree = cKDTree(ref_positions)
        # 对每一个ref_scene的pose找距离前50近的merge_scene的pose
        distances, closest_indices = ref_tree.query(merge_positions, k=50)
        distances_flat = distances.flatten()
        # 对距离进行排序
        sorted_indices = np.argsort(distances_flat)
        # print(distances_flat[sorted_indices[0]])
        if distances_flat[sorted_indices[0]] > 10:
            return -1
        # 计算前20对姿势数据之间的平均距离
        average_distance = np.mean(distances_flat[sorted_indices[:20]])
        return average_distance

    # 得到两个场景中sfm_pose_INTER_aligned_time_TUM.txt
    # 遍历一个场景的所有帧，找另一个场景中距离其最近的帧
    # 按照距离打分
    # 返回平均距离
    def score_scenes(self, scene_id1, scene_id2):
        if scene_id1 == scene_id2:
            return 0
        scene1 = join(self.base_dir, scene_id1)
        scene2 = join(self.base_dir, scene_id2)
        ref_scene_pose = self.get_scene_pose(scene1)
        merge_scene_pose = self.get_scene_pose(scene2)
        ref_positions = ref_scene_pose[:, :3, 3]
        merge_positions = merge_scene_pose[:, :3, 3]
        ref_tree = cKDTree(ref_positions)
        # 对每一个ref_scene的pose找距离最近的merge_scene的pose
        # 并计算距离和打分
        distances, closest_indices = ref_tree.query(merge_positions, k=1)
        score = np.sum(distances <= 1) * 10 + np.sum((distances > 1) & (distances <= 5)) * 5 + np.sum(
            (distances > 5) & (distances <= 10)) * 2
        return score

    # 得到两个场景中sfm_pose_INTER_aligned_time_TUM.txt中最接近的
    # 数据的索引（表示其在TUM文件中的行数，需要利用INDEX来转换到实际上对应aligned文件中图片的行数）
    # 返回两个list，分别表示ref和merge场景中最接近的TUM数据的索引
    def get_closest_index(self):
        ref_scene_pose = self.get_scene_pose(self.ref_scene)
        merge_scene_pose = self.get_scene_pose(self.merge_scene)
        ref_positions = ref_scene_pose[:, :3, 3]
        merge_positions = merge_scene_pose[:, :3, 3]
        ref_tree = cKDTree(ref_positions)
        closest_pairs = []
        close_ref_scene_index = []
        close_merge_scene_index = []
        # 对每一个ref_scene的pose找距离前50近的merge_scene的pose
        distances, closest_indices = ref_tree.query(merge_positions, k=50)
        # print(distances)
        distances_flat = distances.flatten()
        # print(distances_flat)
        # closest_indices_flat 排列成（i,j）的形式
        closest_indices_flat = closest_indices.flatten()
        for i in range(len(closest_indices_flat)):
            closest_pairs.append((int(i / 50), closest_indices_flat[i]))
        closest_pairs = np.array(closest_pairs)
        # print(closest_pairs)
        # 对距离进行排序
        sorted_indices = np.argsort(distances_flat)
        print(distances_flat[sorted_indices[0]])
        '''if distances_flat[sorted_indices[0]] > 10:
            return False'''
        sorted_closest_pairs = closest_pairs[sorted_indices]
        for pair in sorted_closest_pairs:
            # print(pair)
            if pair[0] in close_merge_scene_index or pair[1] in close_ref_scene_index:
                continue
            if len(close_ref_scene_index) >= 50:
                break
            close_ref_scene_index.append(pair[1])
            close_merge_scene_index.append(pair[0])

        close_ref_scene_index = np.array(close_ref_scene_index)
        print(close_ref_scene_index)
        close_merge_scene_index = np.array(close_merge_scene_index)
        sorted_close_ref_scene_index_indices = np.argsort(close_ref_scene_index)
        sorted_close_ref_scene_index = close_ref_scene_index[sorted_close_ref_scene_index_indices]
        sorted_close_merge_scene_index = close_merge_scene_index[sorted_close_ref_scene_index_indices]
        print(sorted_close_ref_scene_index)
        # print(sorted_close_ref_scene_index)
        # print(sorted_close_merge_scene_index)
        # 判断sorted_close_ref_scene_index是否连着
        temp_i = 0
        continous = True
        for i in range(len(sorted_close_ref_scene_index)):
            if i == 0:
                continue
            # 如果目前的所引与上一个索引离得很远
            if sorted_close_ref_scene_index[i] - sorted_close_ref_scene_index[i - 1] > 50:
                #如果中间能够剩余30帧，则直接将这一段作为选择的帧
                if i - temp_i > 30:
                    print("sorted_close_ref_scene_index is not continuous!")
                    chosen_ref_index = range(sorted_close_ref_scene_index[temp_i], sorted_close_ref_scene_index[i - 1])
                    chosen_merge_index = sorted_close_merge_scene_index[temp_i:i - 1]
                    merge_index_min = min(chosen_merge_index)
                    merge_index_max = max(chosen_merge_index)
                    chosen_merge_index = range(merge_index_min, merge_index_max)
                    print(chosen_ref_index)
                    print(chosen_merge_index)
                    return chosen_ref_index, chosen_merge_index
                #如果中间不能剩余30帧，则再去寻找下一帧
                else:
                    temp_i = i
                    continous = False
            #如果到了最后一帧，且不连续，且中间帧数大于30，则直接将这一段作为选择的帧
            if i == len(sorted_close_ref_scene_index) - 1 and not continous and i - temp_i > 30:
                print("sorted_close_ref_scene_index is not continuous!")
                chosen_ref_index = range(sorted_close_ref_scene_index[temp_i], sorted_close_ref_scene_index[i])
                chosen_merge_index = sorted_close_merge_scene_index[temp_i:i]
                merge_index_min = min(chosen_merge_index)
                merge_index_max = max(chosen_merge_index)
                chosen_merge_index = range(merge_index_min, merge_index_max)
                print(chosen_ref_index)
                print(chosen_merge_index)
                return chosen_ref_index, chosen_merge_index
        print("sorted_close_ref_scene_index is continuous!")
        return range(min(sorted_close_ref_scene_index), max(sorted_close_ref_scene_index)), range(
            min(sorted_close_merge_scene_index), max(sorted_close_merge_scene_index))

    # 生成cameras.txt文件
    # 先复制ref_scene的cameras.txt文件到
    # 再将merge_scene的cameras.txt文件的内容加到后面
    # 通过增加ref_cameras_count来保证camera_id的唯一性
    def make_cameras_txt(self):
        ref_cameras_txt_path = join(self.ref_scene, 'sfm', 'chouzhen', 'cameras.txt')
        merge_cameras_txt_path = join(self.merge_scene, 'sfm', 'chouzhen', 'cameras.txt')
        output_cameras_txt_path = join(self.sfm_dir, 'cameras.txt')
        with open(ref_cameras_txt_path, 'r') as ref:
            ref_cameras_lines = ref.readlines()
        ref_cameras = set()
        for ref_cameras_line in ref_cameras_lines:
            elements = ref_cameras_line.strip().split()
            if len(elements) > 0:
                camera_id = elements[0]
                ref_cameras.add(camera_id)
        ref_cameras_count = len(ref_cameras)
        with open(merge_cameras_txt_path, 'r') as merge:
            merge_cameras_lines = merge.readlines()
        with open(output_cameras_txt_path, 'w') as output:
            for ref_cameras_line in ref_cameras_lines:
                output.write(ref_cameras_line)
            for merge_cameras_line in merge_cameras_lines:
                elements = merge_cameras_line.strip().split()
                if len(elements) > 0:
                    camera_id = str(int(elements[0]) + ref_cameras_count)
                    modified_line = str(camera_id) + ' ' + ' '.join(elements[1:]) + '\n'
                    output.write(modified_line)

    # 得到未经变换的相机相对车身的外参
    # 先保存ref_scene的cam_pose.txt
    # 再保存merge_scene的cam_pose.txt
    def make_ref_cam_pose_txt(self):
        destination_file_path = join(self.sfm_dir, 'ref_cam_pose.txt')
        with open(destination_file_path, 'w') as f:
            for i, camera_name in enumerate(self.camera_names):
                # print(camera_name)
                extrinsics = self.ref_extrinsics[camera_name]
                rotation_matrix = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]
                r = Rotation.from_matrix(rotation_matrix)
                rquat = r.as_quat()
                qw, qx, qy, qz = rquat[3], rquat[0], rquat[1], rquat[2]
                fname = 'ref_' + camera_name + '.jpg'
                f.write(
                    f'{i + 1} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} {i + 1} {fname}\n\n')
            for j, camera_name in enumerate(self.camera_names):
                # print(camera_name)
                extrinsics = self.merge_extrinsics[camera_name]
                rotation_matrix = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]
                r = Rotation.from_matrix(rotation_matrix)
                rquat = r.as_quat()
                qw, qx, qy, qz = rquat[3], rquat[0], rquat[1], rquat[2]
                fname = 'merge_' + camera_name + '.jpg'
                f.write(
                    f'{j + 1 + self.camera_count} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} {j + 1 + self.camera_count} {fname}\n\n')

    # 生成rig.json文件，将两个场景的相机组的数据分别记录
    def make_rig_json(self):
        destination_file_path = join(self.sfm_dir, 'rig.json')
        camera_pose_txt_path = join(self.sfm_dir, 'cam_pose.txt')

        # print(camera_pose_txt_path)
        with open(camera_pose_txt_path, 'w') as file:
            with open(destination_file_path, 'w') as f:
                ref_json_data = {
                    "ref_camera_id": 1,
                    "cameras": []
                }
                merge_json_data = {
                    "ref_camera_id": 1 + self.camera_count,
                    "cameras": []
                }
                json_data_list = []
                ref_pose_for_ref = self.ref_extrinsics[self.camera_names[0]]
                ref_pose_for_merge = self.merge_extrinsics[self.camera_names[0]]
                for i, camera_name in enumerate(self.camera_names):
                    # print(camera_name)
                    ref_extrinsics = self.ref_extrinsics[camera_name] @ np.linalg.inv(ref_pose_for_ref)  # 相机坐标系下的世界
                    merge_extrinsics = self.merge_extrinsics[camera_name] @ np.linalg.inv(ref_pose_for_merge)
                    ref_rotation_matrix = ref_extrinsics[:3, :3]
                    merge_rotation_matrix = merge_extrinsics[:3, :3]
                    ref_translation = ref_extrinsics[:3, 3]
                    merge_translation = merge_extrinsics[:3, 3]
                    ref_r = Rotation.from_matrix(ref_rotation_matrix)
                    merge_r = Rotation.from_matrix(merge_rotation_matrix)
                    ref_rquat = ref_r.as_quat()
                    merge_rquat = merge_r.as_quat()
                    ref_qw, ref_qx, ref_qy, ref_qz = ref_rquat[3], ref_rquat[0], ref_rquat[1], ref_rquat[2]
                    merge_qw, merge_qx, merge_qy, merge_qz = merge_rquat[3], merge_rquat[0], merge_rquat[1], \
                        merge_rquat[2]
                    ref_fname = 'ref_' + camera_name + '.jpg'
                    merge_fname = 'merge_' + camera_name + '.jpg'
                    file.write(
                        f'{i + 1} {ref_qw} {ref_qx} {ref_qy} {ref_qz} {ref_translation[0]} {ref_translation[1]} {ref_translation[2]} {i + 1} {ref_fname}\n\n')
                    file.write(
                        f'{i + 1 + self.camera_count} {merge_qw} {merge_qx} {merge_qy} {merge_qz} {merge_translation[0]} {merge_translation[1]} {merge_translation[2]} {i + 1 + self.camera_count} {merge_fname}\n\n')
                    ref_pose = {
                        "camera_id": i + 1,
                        "image_prefix": join(self.ref_scene_id,
                                             self.img_data_name,
                                             self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort/'),
                        "rel_tvec": ref_translation.tolist(),
                        "rel_qvec": [ref_qw, ref_qx, ref_qy, ref_qz]
                    }
                    merge_pose = {
                        "camera_id": i + 1 + self.camera_count,
                        "image_prefix": join(self.merge_scene_id,
                                             self.img_data_name,
                                             self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort/'),
                        "rel_tvec": merge_translation.tolist(),
                        "rel_qvec": [merge_qw, merge_qx, merge_qy, merge_qz]
                    }

                    ref_json_data["cameras"].append(ref_pose)
                    merge_json_data["cameras"].append(merge_pose)
                json_data_list.append(ref_json_data)
                json_data_list.append(merge_json_data)
                json.dump(json_data_list, f, indent=4)

    # 生成images_list.txt文件，记录所有被用到了的图片的路径
    # 同时生成ref_images_list.txt文件，记录ref_scene中被用到了的图片的路径
    # ref_images_list.txt中的图片的位姿将保持固定
    def make_images_list(self):
        destination_path = join(self.sfm_dir, 'images_list.txt')
        ref_images_list = join(self.sfm_dir, 'ref_images_list.txt')
        ref_index = get_data_from_index_txt(join(self.ref_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt'))
        merge_index = get_data_from_index_txt(join(self.merge_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt'))
        ref_scene_pose = self.get_scene_pose(self.ref_scene)
        merge_scene_pose = self.get_scene_pose(self.merge_scene)
        ref_range, merge_range = self.get_closest_index()
        ref_used_index = []
        merge_used_index = []
        start = True
        for i in ref_range:
            if start:
                ref_used_index.append(i)
                start = False
                prev_pose = ref_scene_pose[i]
            else:
                distance = np.linalg.norm(ref_scene_pose[i][:3, 3] - prev_pose[:3, 3])
                if distance > self.distance_step:
                    ref_used_index.append(i)
                    prev_pose = ref_scene_pose[i]
        print(ref_used_index)
        start = True
        for i in merge_range:
            if start:
                merge_used_index.append(i)
                start = False
                prev_pose = merge_scene_pose[i]
            else:
                distance = np.linalg.norm(merge_scene_pose[i][:3, 3] - prev_pose[:3, 3])
                if distance > self.distance_step:
                    merge_used_index.append(i)
                    prev_pose = merge_scene_pose[i]
        print(merge_used_index)
        with open(ref_images_list, 'w') as ref_output_file:
            with open(destination_path, 'w') as output_file:
                for camera_name in self.camera_names:
                    ref_txt_path = join(self.ref_scene, self.img_data_name,
                                        self.image_folder_prefix + camera_name + self.image_folder_suffix + "_aligned.txt")
                    with open(ref_txt_path, 'r') as f:
                        ref_image_names = f.readlines()
                    for index in ref_used_index:
                        image_name = ref_image_names[ref_index[index]].replace("\n", "")
                        output_file.write(join(self.ref_scene_id, self.img_data_name,
                                               self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort',
                                               image_name
                                               ) + '\n')
                        ref_output_file.write(join(self.ref_scene_id, self.img_data_name,
                                                   self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort',
                                                   image_name
                                                   ) + '\n')
                for camera_name in self.camera_names:
                    mer_txt_path = join(self.merge_scene, self.img_data_name,
                                        self.image_folder_prefix + camera_name + self.image_folder_suffix + "_aligned.txt")
                    with open(mer_txt_path, 'r') as f:
                        mer_image_names = f.readlines()
                    for index in merge_used_index:
                        # print(index)
                        image_name = mer_image_names[merge_index[index]].replace("\n", "")
                        output_file.write(join(self.merge_scene_id, self.img_data_name,
                                               self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort',
                                               image_name
                                               ) + '\n')

    # 生成car_poses.txt文件，将两个场景的车身位姿写入
    # 里面只用TUM文件中用到的位姿
    # 满足clomap格式(从1开始)，但是没有对位姿做逆变换
    def make_car_pose_txt(self):
        # print(scene)
        destination_file_path = join(self.sfm_dir, 'car_poses.txt')
        ref_poses = self.get_scene_pose(self.ref_scene)
        merge_poses = self.get_scene_pose(self.merge_scene)
        ref_index = get_data_from_index_txt(join(self.ref_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt'))
        merge_index = get_data_from_index_txt(join(self.merge_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt'))
        count = 1
        with open(destination_file_path, 'w') as f:
            for i, ref_pose in enumerate(ref_poses):  # type: ignore
                r = Rotation.from_matrix(ref_pose[:3, :3])
                t = ref_pose[:3, 3]
                rquat = r.as_quat()
                index = ref_index[i]
                fname = str(i) + '.jpg'
                f.write(f'{count} {rquat[3]} {rquat[0]} {rquat[1]} {rquat[2]} {t[0]} {t[1]} {t[2]} {1} {fname}\n\n')
                count += 1
            self.ref_pose_count = count
            for i, merge_pose in enumerate(merge_poses):  # type: ignore
                r = Rotation.from_matrix(merge_pose[:3, :3])
                t = merge_pose[:3, 3]
                rquat = r.as_quat()
                index = merge_index[i]
                fname = str(i + self.ref_pose_count - 1) + '.jpg'
                f.write(f'{count} {rquat[3]} {rquat[0]} {rquat[1]} {rquat[2]} {t[0]} {t[1]} {t[2]} {1} {fname}\n\n')
                count += 1
            self.merge_pose_count = count - self.ref_pose_count

    # 生成index.txt文件，记录所有图片的索引
    # 只记录了TUM文件中用到的位姿，可能相比于原scene会少最后几帧
    # 但是数量是和car_poses.txt对应的
    # 从0开始
    def make_index_txt(self):
        ref_aligned_txt = join(self.ref_scene, self.img_data_name, 'aligned_time.txt')
        merge_aligned_txt = join(self.merge_scene, self.img_data_name, 'aligned_time.txt')
        ref_sfm_index = join(self.ref_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt')
        merge_sfm_index = join(self.merge_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt')
        with open(ref_sfm_index, 'r') as ref_index_file:
            ref_index_lines = ref_index_file.readlines()
        ref_used_index = range(int(ref_index_lines[0]), 1 + int(ref_index_lines[-1]))
        with open(merge_sfm_index, 'r') as merge_index_file:
            merge_index_lines = merge_index_file.readlines()
        merge_used_index = range(int(merge_index_lines[0]), 1 + int(merge_index_lines[-1]))
        output_txt = join(self.sfm_dir, 'index.txt')
        time_index = {}
        # 记录aligned_time.txt中的时间，只记录了TUM文件中用到的时刻
        with open(ref_aligned_txt, 'r') as ref_aligned:
            time_index['aligned'] = []
            lines = ref_aligned.readlines()
            for i in ref_used_index:
                parts = lines[i].strip().split()
                time_index['aligned'].append(parts[0])
            '''for i, line in enumerate(lines):
                parts = line.strip().split()
                if i < self.ref_pose_count:
                    time_index['aligned'].append(parts[0])
                else:
                    # print(i, self.ref_pose_count)
                    break'''
        with open(merge_aligned_txt, 'r') as merge_aligned:
            lines = merge_aligned.readlines()
            for j in merge_used_index:
                parts = lines[j].strip().split()
                time_index['aligned'].append(parts[0])
            '''for j, line in enumerate(lines):
                parts = line.strip().split()
                if j < self.merge_pose_count:
                    time_index['aligned'].append(parts[0])
                else:
                    # print(j, self.merge_pose_count)
                    break'''
        # 记录每个相机的时间，只记录了TUM文件中用到的时刻
        for camera_name in self.camera_names:
            # print(camera_name)
            ref_txt_path = join(self.ref_scene, self.img_data_name,
                                self.image_folder_prefix + camera_name + self.image_folder_suffix + '_aligned.txt')
            merge_txt_path = join(self.merge_scene, self.img_data_name,
                                  self.image_folder_prefix + camera_name + self.image_folder_suffix + '_aligned.txt')

            time_index[camera_name] = []
            ref_folder_name = join(self.ref_scene_id, self.img_data_name,
                                   self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort/')
            merge_folder_name = join(self.merge_scene_id, self.img_data_name,
                                     self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort/')

            with open(ref_txt_path, 'r') as f:
                lines = f.readlines()
                for i in ref_used_index:
                    parts = lines[i].strip().split()
                    time_index[camera_name].append(ref_folder_name + parts[0])

            with open(merge_txt_path, 'r') as f:
                lines = f.readlines()
                for j in merge_used_index:
                    parts = lines[j].strip().split()
                    time_index[camera_name].append(merge_folder_name + parts[0])

        with open(output_txt, 'w') as f:
            for i in range(len(time_index['aligned'])):
                f.write(str(i) + ' ')
                f.write(time_index['aligned'][i])
                for camera_name in self.camera_names:
                    f.write(' ' + time_index[camera_name][i])
                f.write('\n')

    # 按照images_list.txt里面的图片名，生成images.txt
    # 按照camera_id的顺序排列
    def make_images_txt(self):
        count = 1
        destination_file_path = join(self.sfm_dir, 'images.txt')
        list_path = join(self.sfm_dir, "images_list.txt")
        index_path = join(self.sfm_dir, "index.txt")
        car_poses_path = join(self.sfm_dir, "car_poses.txt")
        poses = []
        with open(car_poses_path, 'r') as f:
            lines = f.readlines()[::2]
        for line in lines:
            parts = line.strip().split()
            qw, qx, qy, qz, tx, ty, tz = [float(i) for i in parts[1:8]]
            q = Quaternion(qw, qx, qy, qz)
            r = q.rotation_matrix
            t = np.array([tx, ty, tz])
            pose = np.eye(4)
            pose[:3, :3] = r
            pose[:3, 3] = t
            poses.append(pose)
        index = {}
        with open(index_path, 'r') as index_file:
            lines = index_file.readlines()
        # 求出该图片名对应的索引index，它对应car_poses.txt中的第index行数据（通过[::2]读取）
        for line in lines:
            parts = line.strip().split()
            for i in range(self.camera_count):
                index[parts[i + 2]] = parts[0]
        with open(list_path, 'r') as file:
            fnames = [line.strip() for line in file.readlines()]
        with open(destination_file_path, 'w') as f:
            for fname in fnames:
                for j, camera_name in enumerate(self.camera_names):
                    image_floder_name = self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort'
                    if image_floder_name not in fname or self.merge_scene_id in fname:
                        continue
                    extrinsic = self.ref_extrinsics[camera_name]
                    add = 0
                    pose = poses[int(index[fname])]
                    pose = pose @ np.linalg.inv(extrinsic)
                    Ro = np.linalg.inv(pose[:3, :3])
                    T = -np.matmul(Ro, pose[:3, 3])
                    r = Rotation.from_matrix(Ro[:3, :3])
                    rquat = r.as_quat()
                    f.write(
                        f'{count} {rquat[3]} {rquat[0]} {rquat[1]} {rquat[2]} {T[0]} {T[1]} {T[2]} {j + 1 + add} {fname}\n\n')
                    count += 1
            for fname in fnames:
                for j, camera_name in enumerate(self.camera_names):
                    image_floder_name = self.image_folder_prefix + camera_name + self.image_folder_suffix + '-undistort'
                    if image_floder_name not in fname or self.ref_scene_id in fname:
                        continue
                    extrinsic = self.merge_extrinsics[camera_name]
                    add = self.camera_count
                    pose = poses[int(index[fname])]
                    pose = pose @ np.linalg.inv(extrinsic)
                    Ro = np.linalg.inv(pose[:3, :3])
                    T = -np.matmul(Ro, pose[:3, 3])
                    r = Rotation.from_matrix(Ro[:3, :3])
                    rquat = r.as_quat()
                    f.write(
                        f'{count} {rquat[3]} {rquat[0]} {rquat[1]} {rquat[2]} {T[0]} {T[1]} {T[2]} {j + 1 + add} {fname}\n\n')
                    count += 1

    # 处理trans.txt
    # 将merge_scene里面的稠密地面点云做变换
    # 保存到sfm_dir里面
    def transform_points3D(self):
        # 若使用acmp的稠密地面点云，需要修改merge_ply_path
        # merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP','acmap_road_ransac.ply')
        merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'acmap_road_ransac.ply')
        ref_ply_path = join(self.ref_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'acmap_road_ransac.ply')
        transed_merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP',
                                      'transed_acmap_road_ransac.ply')
        transed_ref_ply_path = join(self.ref_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'transed_acmap_road_ransac.ply')
        shutil.copy(ref_ply_path, transed_ref_ply_path)
        trans_txt_path = join(self.sfm_dir, 'merge_mapper', 'trans.txt')

        or_ref_tum_path = join(self.ref_scene, 'files', 'sfm_pose_INTER_aligned_time_TUM.txt')
        or_ref_index_path = join(self.ref_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt')
        or_merge_tum_path = join(self.merge_scene, 'files', 'sfm_pose_INTER_aligned_time_TUM.txt')
        or_merge_index_path = join(self.merge_scene, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt')
        or_merge_poses, merge_time_stamp = self.loadarray_hozon(np.loadtxt(or_merge_tum_path))
        transed_ref_tum_path = join(self.ref_scene, 'files', 'transed_sfm_pose_INTER_aligned_time_TUM.txt')
        transed_ref_index_path = join(self.ref_scene, 'files', 'transed_sfm_pose_INTER_aligned_time_INDEX.txt')
        transed_merge_tum_path = join(self.merge_scene, 'files', 'transed_sfm_pose_INTER_aligned_time_TUM.txt')
        transed_merge_index_path = join(self.merge_scene, 'files', 'transed_sfm_pose_INTER_aligned_time_INDEX.txt')
        shutil.copy(or_merge_index_path, transed_merge_index_path)
        shutil.copy(or_ref_index_path, transed_ref_index_path)
        shutil.copy(or_ref_tum_path, transed_ref_tum_path)
        with open(trans_txt_path, 'r') as f:
            line = f.readline()
        parts = line.strip().split()
        qw, qx, qy, qz, tx, ty, tz = [float(i) for i in parts]
        q = Quaternion(qw, qx, qy, qz)
        r = q.rotation_matrix
        t = np.array([tx, ty, tz])
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t
        inv_trans = np.linalg.inv(trans)
        transed_merge_poses = or_merge_poses @ inv_trans
        with open(transed_merge_tum_path, 'w') as f:
            for (pose, time) in zip(transed_merge_poses, merge_time_stamp):
                quat = Quaternion(matrix=pose[:3, :3])
                tx, ty, tz = pose[:3, 3]
                f.write(f'{time} {tx} {ty} {tz} {quat.x} {quat.y} {quat.z} {quat.w}\n')
        r_inv = inv_trans[:3, :3]
        t_inv = inv_trans[:3, 3]
        # 求逆变换
        pcd = o3d.io.read_point_cloud(merge_ply_path)
        pcd_points = np.asarray(pcd.points)
        pcd_points = pcd_points @ r_inv.T + t_inv
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        o3d.io.write_point_cloud(transed_merge_ply_path, pcd)

    def transform_sparse_points3D(self):
        # 若使用acmp的稠密地面点云，需要修改merge_ply_path
        # merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP','acmap_road_ransac.ply')
        # ref_ply_path = join(self.ref_scene, 'sfm', 'chouzhen', 'rig_mapper','sparse.ply')
        merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'rig_mapper', 'sparse.ply')
        get_sparse_ply(self.ref_scene)
        get_sparse_ply(self.merge_scene)
        trans_txt_path = join(self.sfm_dir, 'merge_mapper', 'trans.txt')
        new_ply_path = join(self.sfm_dir, 'merge_sparse_transed.ply')
        with open(trans_txt_path, 'r') as f:
            line = f.readline()
        parts = line.strip().split()
        qw, qx, qy, qz, tx, ty, tz = [float(i) for i in parts]
        q = Quaternion(qw, qx, qy, qz)
        r = q.rotation_matrix
        t = np.array([tx, ty, tz])
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t
        inv_trans = np.linalg.inv(trans)
        r_inv = inv_trans[:3, :3]
        t_inv = inv_trans[:3, 3]
        # 求逆变换
        pcd = o3d.io.read_point_cloud(merge_ply_path)
        pcd_points = np.asarray(pcd.points)
        pcd_points = pcd_points @ r_inv.T + t_inv
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        o3d.io.write_point_cloud(new_ply_path, pcd)

    def transform_dense_points3D(self):
        # 若使用acmp的稠密地面点云，需要修改merge_ply_path
        # merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP','acmap_road_ransac.ply')
        ref_ply_path = join(self.ref_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'acmap_all_fil.ply')
        merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'acmap_all_fil.ply')
        ref_new_ply_path = join(self.ref_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'transed_acmap_all_fil.ply')
        shutil.copy(ref_ply_path, ref_new_ply_path)
        trans_txt_path = join(self.sfm_dir, 'merge_mapper', 'trans.txt')
        new_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP', 'transed_acmap_all_fil.ply')
        with open(trans_txt_path, 'r') as f:
            line = f.readline()
        parts = line.strip().split()
        qw, qx, qy, qz, tx, ty, tz = [float(i) for i in parts]
        q = Quaternion(qw, qx, qy, qz)
        r = q.rotation_matrix
        t = np.array([tx, ty, tz])
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t
        inv_trans = np.linalg.inv(trans)
        r_inv = inv_trans[:3, :3]
        t_inv = inv_trans[:3, 3]
        # 求逆变换
        pcd = o3d.io.read_point_cloud(merge_ply_path)
        pcd_points = np.asarray(pcd.points)
        pcd_points = pcd_points @ r_inv.T + t_inv
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        o3d.io.write_point_cloud(new_ply_path, pcd)

    def make_points3D_txt(self):
        txt_path = join(self.sfm_dir, 'points3D.txt')
        with open(txt_path, 'w') as f:
            pass

    def process_trans_txt(self):
        trans_txt_path = join(self.base_dir, 'trans.txt')
        source_trans_txt_path = join(self.sfm_dir, 'merge_mapper', 'trans.txt')
        if not os.path.exists(source_trans_txt_path):
            return
        with open(source_trans_txt_path, 'r') as f:
            line = f.readline()
        with  open(trans_txt_path, 'a') as f:
            f.write(self.ref_scene_id + ' ' + self.merge_scene_id + ' ' + line)

    # 将所有场景变换到一个坐标系内
    # 处理点云和TUM文件
    def get_all_scene_in_one_coord(self):
        trans = {}
        trans_txt_path = join(self.base_dir, 'trans.txt')
        with open(trans_txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            qw, qx, qy, qz, tx, ty, tz = [float(i) for i in parts[2:]]
            q = Quaternion(qw, qx, qy, qz)
            r = q.rotation_matrix
            t = np.array([tx, ty, tz])
            trans_matrix = np.eye(4)
            trans_matrix[:3, :3] = r
            trans_matrix[:3, 3] = t
            inv_trans_matrix = np.linalg.inv(trans_matrix)
            trans[parts[0], parts[1]] = inv_trans_matrix
        #print(trans)
        scene_ids = [scene_id for pair in trans.keys() for scene_id in pair]
        scene_id_counter = Counter(scene_ids)
        # 找到出现次数最多的场景 ID，它作为基准场景
        most_common_scene_id = scene_id_counter.most_common(1)[0][0]
        print(most_common_scene_id)
        rts = {}
        used_pairs = []
        # 找到直接与基准场景相关的相对变换
        for pair, rt in trans.items():
            if pair[0] == most_common_scene_id:
                print(pair)
                rts[pair[1]] = rt
                used_pairs.append(pair)
            elif pair[1] == most_common_scene_id:
                print(pair)
                rts[pair[0]] = np.linalg.inv(rt)
                used_pairs.append(pair)
        # 更新相对变换，移除已经使用的相对变换
        for pair in used_pairs:
            trans.pop(pair)
        # 如果还有场景没有被使用，则使用相对变换
        while trans:
            used_pairs = []
            used_scene_ids = []
            used_rts = []
            for pair, rt in trans.items():
                for scene_id, rel_rt in rts.items():
                    # 如果 pair[0] 等于 scene_id，则使用 pair[1] 相对于 scene_id 的相对变换
                    if pair[0] == scene_id:
                        print(pair, scene_id)
                        used_pairs.append(pair)
                        used_scene_ids.append(pair[1])
                        used_rts.append(rel_rt @ rt)
                    # 如果 pair[1] 等于 scene_id，则使用 pair[0] 相对于 scene_id 的相对变换
                    elif pair[1] == scene_id:
                        print(pair, scene_id)
                        used_pairs.append(pair)
                        used_scene_ids.append(pair[0])
                        used_rts.append(rel_rt @ np.linalg.inv(rt))
            # 更新相对变换
            for pair in used_pairs:
                trans.pop(pair)
            for (used_scene_id, used_rt) in zip(used_scene_ids, used_rts):
                rts[used_scene_id] = used_rt
        #将稀疏点云合并
        merged_ply_folder = join(self.base_dir, 'merged_ply')
        if not os.path.exists(merged_ply_folder):
            os.makedirs(merged_ply_folder)
        for scene_id in self.scene_ids:
            source_tum_path = join(self.base_dir, scene_id, 'files', 'sfm_pose_INTER_aligned_time_TUM.txt')
            source_index_path = join(self.base_dir, scene_id, 'files', 'sfm_pose_INTER_aligned_time_INDEX.txt')
            target_tum_path = join(self.base_dir, scene_id, 'files', 'transed_sfm_pose_INTER_aligned_time_TUM.txt')
            target_index_path = join(self.base_dir, scene_id, 'files', 'transed_sfm_pose_INTER_aligned_time_INDEX.txt')
            source_ply_path = join(self.base_dir, scene_id, 'sfm', 'chouzhen', 'rig_mapper', 'sparse.ply')
            #source_ply_path = join(self.base_dir, scene_id, 'sfm', 'chouzhen', 'dense_ACMP', 'acmap_road_ransac.ply')
            target_ply_path = join(merged_ply_folder, scene_id + '.ply')
            shutil.copy(source_index_path, target_index_path)
            if scene_id in rts:
                rt = rts[scene_id]
                pcd = o3d.io.read_point_cloud(source_ply_path)
                pcd.transform(rt)
                o3d.io.write_point_cloud(target_ply_path, pcd)
                or_poses, merge_time_stamp = self.loadarray_hozon(np.loadtxt(source_tum_path))
                transed_merge_poses = or_poses @ rt
                with open(target_tum_path, 'w') as f:
                    for (pose, time) in zip(transed_merge_poses, merge_time_stamp):
                        quat = Quaternion(matrix=pose[:3, :3])
                        tx, ty, tz = pose[:3, 3]
                        f.write(f'{time} {tx} {ty} {tz} {quat.x} {quat.y} {quat.z} {quat.w}\n')
            else:
                shutil.copy(source_tum_path, target_tum_path)
                shutil.copy(source_ply_path, target_ply_path)

        return trans

    def transform_sparse_points3D_for_folder(self):
        # 若使用acmp的稠密地面点云，需要修改merge_ply_path
        # merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'dense_ACMP','acmap_road_ransac.ply')
        # ref_ply_path = join(self.ref_scene, 'sfm', 'chouzhen', 'rig_mapper','sparse.ply')
        merge_ply_path = join(self.merge_scene, 'sfm', 'chouzhen', 'rig_mapper', 'sparse.ply')
        get_sparse_ply(self.ref_scene)
        get_sparse_ply(self.merge_scene)
        trans_txt_path = join(self.sfm_dir, 'merge_mapper', 'trans.txt')
        new_ply_path = join(self.sfm_dir, 'merge_sparse_transed.ply')
        with open(trans_txt_path, 'r') as f:
            line = f.readline()
        parts = line.strip().split()
        qw, qx, qy, qz, tx, ty, tz = [float(i) for i in parts]
        q = Quaternion(qw, qx, qy, qz)
        r = q.rotation_matrix
        t = np.array([tx, ty, tz])
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t
        inv_trans = np.linalg.inv(trans)
        r_inv = inv_trans[:3, :3]
        t_inv = inv_trans[:3, 3]
        # 求逆变换
        pcd = o3d.io.read_point_cloud(merge_ply_path)
        pcd_points = np.asarray(pcd.points)
        pcd_points = pcd_points @ r_inv.T + t_inv
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        o3d.io.write_point_cloud(new_ply_path, pcd)

    def make_global_input(self):
        self.make_cameras_txt()
        self.make_ref_cam_pose_txt()
        self.make_rig_json()
        self.make_images_list()
        self.make_car_pose_txt()
        self.make_index_txt()
        self.make_images_txt()
        self.make_points3D_txt()
        # get_video(scene)

    #处理两两场景之间的merge
    def process(self):
        self.success = os.path.exists(join(self.sfm_dir, 'merge_mapper', '0', 'images.txt'))
        #如果之前重建过了
        if self.success:
            self.transform_sparse_points3D()
            self.transform_points3D()
            return True
        self.make_global_input()
        last_slash_index = self.sfm_dir.rfind('/sfm')
        if last_slash_index != -1:
            folder = self.sfm_dir[:last_slash_index] + self.sfm_dir[last_slash_index:].replace('/sfm', '')
        print(folder)
        get_db(folder)
        merge_mapper(folder)
        self.success = os.path.exists(join(self.sfm_dir, 'merge_mapper', '0', 'images.txt'))
        # self.success = True
        if self.success:
            # self.transform_points3D()
            self.transform_sparse_points3D()
            # self.transform_dense_points3D()
        return True

    # 处理一个文件夹
    def process_folders(self):
        has_couples = np.any(self.done_scenes == 0)
        # 还有场景没有连通
        while has_couples:
            i, j = self.make_yaml_path()
            self.process()
            #self.success = True
            if self.success:
                self.done_scenes[i, j] = 100
                self.done_scenes[j, i] = 200
                check_array(self.done_scenes)
            print(self.done_scenes)
            has_couples = np.any(self.done_scenes == 0)
            self.process_trans_txt()
        # 处理完所有场景
        mydataset.get_all_scene_in_one_coord()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='merge_sfm.yaml')
    args = parser.parse_args()
    mydataset = MergeInput(args.yaml_path)
    # mydataset.make_yaml_path()
    mydataset.process_folders()

    '''mydataset.score_scenes('EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-050843_54',
                           'EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-013133_23')'''
    # mydataset.transform_points3D()
    # mydataset.transform_dense_points3D()
    # mydataset.inv_transform_sparse_points3D()
