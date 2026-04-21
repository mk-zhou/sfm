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
from tqdm import tqdm
import sys

sys.path.append(('./super_colmap'))
sys.path.append(('./utils'))
from super_colmap.super_colmap import superpoint_geomatch
from step import rig_mapper
import argparse


def check_file_existence(file_path):
    if not isfile(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在。")

# 生成database.db
def get_db(scene):
    turn = os.path.exists(join(scene, 'sfm', 'chouzhen', 'turn_point_id.txt'))
    if turn:
        superpoint_geomatch(scene, 'sfm', 2)
    else:
        superpoint_geomatch(scene, 'sfm', 0)


class NeZhaDataset:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        super().__init__()
        self.output_dir = configs["output_dir"]
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.distance_step = configs["distance_step"]
        self.width = configs["image_width"]
        self.height = configs["image_height"]
        self.rename_prefix = configs["rename_prefix"]
        self.yaml_path = yaml_path
        self.image_dir = configs["image_dir"]
        self.seg_dir = configs["seg_dir"]
        self.pose_dir = configs["pose_dir"]
        self.camera_names = configs["camera_names"]
        self.all = configs["all"]
        self.intrinsics = {}
        self.extrinsics = {}
        if self.all:
            self.scene_ids = sorted(os.listdir(self.seg_dir))
        else:
            self.scene_ids = configs["scene_ids"]
        for scene_id in self.scene_ids:
            self.intrinsics[scene_id] = {}
            self.extrinsics[scene_id] = {}
            calib_txt_path = join(self.image_dir, scene_id, 'calib.txt')
            with open(calib_txt_path, 'r') as file:
                iPx = file.readlines()
            # homography matrix
            iP0 = np.array([float(i) for i in iPx[0].strip('\n').split(' ')[1:]]).reshape(3, 4)
            iP2 = np.array([float(i) for i in iPx[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
            iP3 = np.array([float(i) for i in iPx[3].strip('\n').split(' ')[1:]]).reshape(3, 4)
            # 内参
            K2 = iP2[:3, :3]
            K3 = iP3[:3, :3]
            self.intrinsics[scene_id][self.camera_names[0]] = K2
            self.intrinsics[scene_id][self.camera_names[1]] = K3
            self.extrinsics[scene_id][self.camera_names[0]] = np.eye(4)
            self.extrinsics[scene_id][self.camera_names[1]] = np.eye(4)
            # calculate real translation
            K2_inv = np.linalg.inv(K2)
            K3_inv = np.linalg.inv(K3)
            T2 = K2_inv @ (iP2[:3, 3].reshape(3, 1)).flatten()
            T3 = K3_inv @ (iP3[:3, 3].reshape(3, 1)).flatten()
            self.extrinsics[scene_id][self.camera_names[0]][:3, 3] = T2
            self.extrinsics[scene_id][self.camera_names[1]][:3, 3] = T3

    # 生成params文件夹
    def get_params_folder(self, scene):
        if not os.path.exists(join(scene, 'sfm', 'chouzhen', 'params')):
            shutil.copytree(self.camera_params_path, join(scene, 'sfm', 'chouzhen', 'params'))

    # 重命名之前重建的结果
    def rename(self, scene_id):
        name = self.rename_prefix
        scene = join(self.output_dir, scene_id)
        rig_mapper_path = join(scene, 'sfm', 'chouzhen', 'rig_mapper')
        if os.path.exists(rig_mapper_path):
            os.rename(rig_mapper_path, join(scene, 'sfm', 'chouzhen', name+'_rig_mapper'))


    def loadarray_hozon(self, array):
        input_pose = array
        # print(input_pose)
        transforms = []
        for i, pose in enumerate(input_pose):
            # 创建4x4的数组并填充
            matrix_4x4 = np.zeros((4, 4))
            matrix_4x4[:3, :4] = pose.reshape(3, 4)  # 将3x4部分填入4x4
            matrix_4x4[3, :] = [0, 0, 0, 1]  # 添加最后一行
            transforms.append(matrix_4x4)
        transforms = np.array(transforms)
        return transforms

    # 得到该场景下的车身位姿
    def get_ref_poses(self, scene_id):
        pose_txt_path = os.path.join(self.pose_dir, scene_id + '.txt')
        car_poses = self.loadarray_hozon(np.loadtxt(pose_txt_path))
        scene = join(self.output_dir, scene_id)
        if not os.path.exists(join(scene)):
            os.makedirs(join(scene))
        if not os.path.exists(join(scene, 'sfm')):
            os.makedirs(join(scene, 'sfm'))
        return car_poses

    # 生成car_poses.txt记录车身位姿
    def get_car_poses(self, scene_id):
        count = 1
        poses = self.get_ref_poses(scene_id)
        destination_file_path = join(self.output_dir, scene_id, 'sfm', 'car_poses.txt')
        with open(destination_file_path, 'w') as f:
            for i, pose in enumerate(poses):  # type: ignore
                r = Rotation.from_matrix(pose[:3, :3])
                t = pose[:3, 3]
                rquat = r.as_quat()
                fname = f"{i:06d}.png"
                f.write(f'{count} {rquat[3]} {rquat[0]} {rquat[1]} {rquat[2]} {t[0]} {t[1]} {t[2]} {1} {fname}\n\n')
                count += 1

    # 得到未经变换的相机相对车身的外参
    def make_ref_cam_pose_txt(self, scene_id):
        destination_file_path = join(self.output_dir, scene_id, 'sfm', 'ref_cam_pose.txt')
        with open(destination_file_path, 'w') as f:
            for i, camera_name in enumerate(self.camera_names):
                # print(camera_name)
                extrinsics = self.extrinsics[scene_id][camera_name]
                rotation_matrix = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]
                r = Rotation.from_matrix(rotation_matrix)
                rquat = r.as_quat()
                qw, qx, qy, qz = rquat[3], rquat[0], rquat[1], rquat[2]
                fname = camera_name + '.jpg'
                f.write(
                    f'{i + 1} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} {i + 1} {fname}\n\n')

    # 生成rig.json记录相机组信息
    def make_rig_json(self, scene_id):
        destination_file_path = join(self.output_dir, scene_id, 'sfm', 'rig.json')
        camera_pose_txt_path = join(self.output_dir, scene_id, 'sfm', 'cam_pose.txt')

        # print(camera_pose_txt_path)
        with open(camera_pose_txt_path, 'w') as file:
            with open(destination_file_path, 'w') as f:
                json_data = {
                    "ref_camera_id": 1,
                    "cameras": []
                }
                json_data_list = []
                ref_pose = self.extrinsics[scene_id][self.camera_names[0]]
                for i, camera_name in enumerate(self.camera_names):
                    # print(camera_name)
                    extrinsics = self.extrinsics[scene_id][camera_name] @ np.linalg.inv(ref_pose)  # 相机坐标系下的世界
                    rotation_matrix = extrinsics[:3, :3]
                    translation = extrinsics[:3, 3]
                    r = Rotation.from_matrix(rotation_matrix)
                    rquat = r.as_quat()
                    qw, qx, qy, qz = rquat[3], rquat[0], rquat[1], rquat[2]
                    fname = camera_name + '.jpg'
                    file.write(
                        f'{i + 1} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} {i + 1} {fname}\n\n')
                    pose = {
                        "camera_id": i + 1,
                        "image_prefix": camera_name,
                        "rel_tvec": translation.tolist(),
                        "rel_qvec": [qw, qx, qy, qz]
                    }
                    json_data["cameras"].append(pose)
                json_data_list.append(json_data)
                json.dump(json_data_list, f, indent=4)

    # 生成cameras.txt记录相机参数
    def make_cameras_txt(self, scene_id):
        i = 0
        txt_path = join(self.output_dir, scene_id, 'sfm', 'cameras.txt')
        with open(txt_path, 'w') as file:
            for camera_id, intrinsics_matrix in self.intrinsics[scene_id].items():
                fx = intrinsics_matrix[0, 0]
                fy = intrinsics_matrix[1, 1]
                cx = intrinsics_matrix[0, 2]
                cy = intrinsics_matrix[1, 2]
                params = [fx, fy, cx, cy]
                # print(camera_id)
                # if camera_id != 'front_wide':
                i = i + 1
                line = f"{i} PINHOLE {self.width} {self.height} {' '.join(map(str, params))}"
                file.write(line + "\n")

    # 生成images_list.txt,记录做重建所需的图片名称
    def make_images_list(self, scene_id):
        destination_path = join(self.output_dir, scene_id, 'sfm', 'images_list.txt')
        root_folder = join(self.output_dir, scene_id)
        sfm_image_folder = join(root_folder, 'image')
        sfm_seg_folder = join(root_folder, 'seg')
        if not os.path.exists(sfm_seg_folder):
            os.makedirs(sfm_seg_folder)
        if not os.path.exists(sfm_image_folder):
            os.makedirs(sfm_image_folder)
        with open(destination_path, 'w') as output_file:
            for camera_name in self.camera_names:
                image_folder = join(self.image_dir, scene_id, camera_name)
                seg_folder = join(self.seg_dir, scene_id, camera_name)
                if not os.path.exists(join(self.output_dir, scene_id, 'image', camera_name)):
                    os.symlink(image_folder, join(self.output_dir, scene_id, 'image', camera_name))
                if not os.path.exists(join(self.output_dir, scene_id, 'seg', camera_name)):
                    os.symlink(seg_folder, join(self.output_dir, scene_id, 'seg', camera_name))
                image_names = sorted([f for f in listdir(image_folder) if isfile(join(image_folder, f))])
                for i, image_name in enumerate(image_names):
                    output_file.write(join(camera_name, image_name) + '\n')

    # 生成images.txt，记录每个图片的位姿
    def make_images_txt(self, scene_id):
        count = 1
        poses = self.get_ref_poses(scene_id)
        destination_file_path = join(self.output_dir, scene_id, 'sfm', 'images.txt')
        list_path = join(self.output_dir, scene_id, "sfm", "images_list.txt")
        with open(list_path, 'r') as file:
            fnames = [line.strip() for line in file.readlines()]
        # print(len(fnames))
        with open(destination_file_path, 'w') as f:
            for j, camera_name in enumerate(self.camera_names):
                extrinsics = self.extrinsics[scene_id][camera_name]
                for i, pose in enumerate(poses):  # type: ignore
                    pose = pose @ np.linalg.inv(extrinsics)
                    Ro = np.linalg.inv(pose[:3, :3])
                    T = -np.matmul(Ro, pose[:3, 3])
                    r = Rotation.from_matrix(Ro[:3, :3])
                    rquat = r.as_quat()
                    f.write(
                        f'{count} {rquat[3]} {rquat[0]} {rquat[1]} {rquat[2]} {T[0]} {T[1]} {T[2]} {j + 1} {fnames[count - 1]}\n\n')
                    count += 1

    # 生成空文件points3D.txt
    def make_points3D_txt(self, scene_id):
        folder_path = join(self.output_dir, scene_id, 'sfm')
        txt_path = join(folder_path, 'points3D.txt')
        with open(txt_path, 'w') as f:
            pass

    # 生成index.txt，以相机组的形式记录图像名称
    def make_index_txt(self, scene_id):
        folder_path = join(self.output_dir, scene_id, 'sfm')
        destination_path = join(folder_path, 'index.txt')
        image_names = {}
        for camera_name in self.camera_names:
            image_folder = join(self.image_dir, scene_id, camera_name)
            image_names[camera_name] = sorted([f for f in listdir(image_folder) if isfile(join(image_folder, f))])
        num_images = len(image_names[self.camera_names[0]])
        with open(destination_path, 'w') as output_file:
            for i in range(num_images):
                my_str = f"{i:06d}.png"
                for camera_name in self.camera_names:
                    my_str += f" {camera_name}/{image_names[camera_name][i]}"
                output_file.write(my_str + '\n')

    # 生成抽帧后的数据，step代表每个多少张图片再取下一张
    def chouzhen(self, scene_id):
        sfm_folder = join(self.output_dir, scene_id, 'sfm')
        print(sfm_folder)
        chouzhen_folder = join(sfm_folder, 'chouzhen')
        images_list_path = join(sfm_folder, 'images_list.txt')
        chouzhen_images_list_path = join(chouzhen_folder, 'images_list.txt')
        camera_pose_txt_path = join(sfm_folder, 'cam_pose.txt')
        chouzhen_camera_pose_txt_path = join(chouzhen_folder, 'cam_pose.txt')
        rig_json_path = join(sfm_folder, 'rig.json')
        chouzhen_rig_json_path = join(chouzhen_folder, 'rig.json')
        ref_cam_pose_txt_path = join(sfm_folder, 'ref_cam_pose.txt')
        chouzhen_ref_cam_pose_txt_path = join(chouzhen_folder, 'ref_cam_pose.txt')
        car_pose_txt_path = join(sfm_folder, 'car_poses.txt')
        chouzhen_car_pose_txt_path = join(chouzhen_folder, 'car_poses.txt')
        points3D_txt_path = join(sfm_folder, 'points3D.txt')
        chouzhen_points3D_txt_path = join(chouzhen_folder, 'points3D.txt')
        cameras_txt_path = join(sfm_folder, 'cameras.txt')
        chouzhen_cameras_txt_path = join(chouzhen_folder, 'cameras.txt')
        images_txt_path = join(sfm_folder, 'images.txt')
        chouzhen_images_txt_path = join(chouzhen_folder, 'images.txt')
        index_txt_path = join(sfm_folder, 'index.txt')
        chouzhen_index_txt_path = join(chouzhen_folder, 'index.txt')
        if not os.path.isdir(chouzhen_folder):
            os.mkdir(chouzhen_folder)
        car_poses = []
        with open(car_pose_txt_path, 'r') as file:
            car_poses_lines = file.readlines()[::2]
        for car_poses_line in car_poses_lines:
            line = car_poses_line.strip().split()
            car_poses.append(np.array([float(i) for i in line[5:8]]))
        used_file_id = []
        for i, car_pose in enumerate(car_poses):
            if i == 0:
                used_file_id.append(i)
                prev_car_pose = car_pose
                continue
            distance = np.linalg.norm(prev_car_pose - car_pose)
            # print(i, prev_car_pose, car_pose)
            # print(distance)
            if distance > self.distance_step:
                used_file_id.append(i)
                prev_car_pose = car_pose
        print(used_file_id)
        with open(images_list_path, 'r') as list:
            fnames = [line.strip() for line in list.readlines()]
        num_snapshot = int(len(fnames) / len(self.camera_names))
        # used_file_id = [i for i in range(num_snapshot) if i % int(self.step) == 0]
        with open(chouzhen_images_list_path, 'w') as file:
            for i in range(len(self.camera_names)):
                for j in used_file_id:
                    # print(i*num_snapshot+j)
                    file.write(fnames[i * num_snapshot + j] + '\n')
        with open(car_pose_txt_path, 'r') as car_pose_txt:
            lines = car_pose_txt.readlines()[::2]
        with open(chouzhen_car_pose_txt_path, 'w') as chouzhen_car_pose_txt:
            for count, j in enumerate(used_file_id):
                parts = lines[j].strip().split()
                parts[0] = str(count + 1)
                chouzhen_car_pose_txt.write(' '.join(parts) + '\n\n')
        count = 1
        with open(images_txt_path, 'r') as images_txt:
            lines = images_txt.readlines()[::2]
        with open(chouzhen_images_txt_path, 'w') as chouzhen_images_txt:
            for i in range(len(self.camera_names)):
                for j in used_file_id:
                    parts = lines[i * num_snapshot + j].strip().split()
                    parts[0] = str(count)
                    chouzhen_images_txt.write(' '.join(parts) + '\n\n')
                    count += 1
        shutil.copy(camera_pose_txt_path, chouzhen_camera_pose_txt_path)
        shutil.copy(rig_json_path, chouzhen_rig_json_path)
        shutil.copy(ref_cam_pose_txt_path, chouzhen_ref_cam_pose_txt_path)
        shutil.copy(points3D_txt_path, chouzhen_points3D_txt_path)
        shutil.copy(cameras_txt_path, chouzhen_cameras_txt_path)
        shutil.copy(index_txt_path, chouzhen_index_txt_path)

    # 生成所需数据
    def make_global_input(self, scene_id):
        self.get_car_poses(scene_id)
        self.make_ref_cam_pose_txt(scene_id)
        self.make_rig_json(scene_id)
        self.make_cameras_txt(scene_id)
        self.make_images_list(scene_id)
        self.make_images_txt(scene_id)
        self.make_points3D_txt(scene_id)
        self.make_index_txt(scene_id)
        self.chouzhen(scene_id)

    # 完成整个工作流程
    def process_scene(self, scene_id):
        scene = join(self.output_dir, scene_id)
        self.make_global_input(scene_id)
        print('make_global_input done')
        shutil.copy(self.yaml_path, join(scene, 'sfm', 'sfm.yaml'))
        self.get_params_folder(scene)
        get_db(scene)
        print('get_db done')
        self.rename(scene_id)
        rig_mapper(scene)
        print('rig_mapper done')


    def process_all(self):
        for scene_id in self.scene_ids:
            self.process_scene(scene_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='odometry_sfm.yaml')
    args = parser.parse_args()
    mydataset = NeZhaDataset(args.yaml_path)
    mydataset.process_all()
    # mydataset.process_scene("04")
