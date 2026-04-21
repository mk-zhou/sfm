import numpy as np
import cv2
import os
from os import listdir
from os.path import join, isfile, isdir
import yaml
import shutil
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import json
from tqdm import tqdm
import sys
import sys
import numpy as np

sys.path.append(('./super_colmap'))
sys.path.append(('./utils'))
from super_colmap.super_colmap import superpoint_geomatch
from step import rig_mapper
import argparse
from nuscenes.nuscenes import NuScenes


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
        self.distance_step = configs["distance_step"]
        self.width = configs["image_width"]
        self.height = configs["image_height"]
        self.rename_prefix = configs["rename_prefix"]
        self.yaml_path = yaml_path
        self.dataset_path = configs["dataset_path"]
        self.camera_names = configs["camera_names"]
        self.all = configs["all"]
        self.intrinsics = {}
        self.extrinsics = {}

        # 初始化 NuScenes 数据集
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.dataset_path, verbose=True)
        # 获取相机信息
        self.scene_ids = []
        for scene in self.nusc.scene[:2]:
            self.scene_ids.append(scene["token"])

        for scene_id in self.scene_ids:
            self.intrinsics[scene_id] = {}
            self.extrinsics[scene_id] = {}
            scene_dict = self.nusc.get('scene', scene_id)
            sample_token = scene_dict['first_sample_token']
            sample = self.nusc.get('sample', sample_token)
            for cam_data_token in sample['data'].values():
                cam_data = self.nusc.get('sample_data', cam_data_token)
                if cam_data['sensor_modality'] == 'camera':
                    for camera_name in self.camera_names:
                        if camera_name == cam_data['channel']:
                            calibrated_sensor_token = cam_data['calibrated_sensor_token']
                            camera_info = self.nusc.get('calibrated_sensor', calibrated_sensor_token)
                            self.intrinsics[scene_id][camera_name] = np.array(camera_info['camera_intrinsic'])
                            camera_quat = [camera_info['rotation'][1], camera_info['rotation'][2],
                                           camera_info['rotation'][3],
                                           camera_info['rotation'][0]]
                            extri_mat = np.eye(4)
                            extri_mat[:3, :3] = Rotation.from_quat(camera_quat).as_matrix()
                            extri_mat[:3, 3] = camera_info['translation']
                            self.extrinsics[scene_id][camera_name] = np.linalg.inv(extri_mat)
        print(self.intrinsics)
        print(self.extrinsics)

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
            os.rename(rig_mapper_path, join(scene, 'sfm', 'chouzhen', name + '_rig_mapper'))

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
        scene_dict = self.nusc.get('scene', scene_id)
        sample_token = scene_dict['first_sample_token']
        car_poses = []
        scene_name = scene_dict["name"]
        scene = join(self.output_dir, scene_name)
        count = 0
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
            cam_front_token = sample['data']['CAM_FRONT']
            cam_front_data = self.nusc.get('sample_data', cam_front_token)
            ego_pose_token = cam_front_data['ego_pose_token']
            ego_pose = self.nusc.get('ego_pose', ego_pose_token)
            ego_pose_mat = np.eye(4)
            ego_quat = [ego_pose['rotation'][1], ego_pose['rotation'][2], ego_pose['rotation'][3],
                        ego_pose['rotation'][0]]
            ego_pose_mat[:3, :3] = Rotation.from_quat(ego_quat).as_matrix()
            ego_pose_mat[:3, 3] = ego_pose['translation']
            if count == 0:
                initinal_pose = ego_pose_mat
                ego_pose_mat = np.linalg.inv(initinal_pose) @ ego_pose_mat
                count = 1
            else:
                ego_pose_mat = np.linalg.inv(initinal_pose) @ ego_pose_mat

            car_poses.append(ego_pose_mat)
        car_poses = np.array(car_poses)
        if not os.path.exists(join(scene)):
            os.makedirs(join(scene))
        if not os.path.exists(join(scene, 'sfm')):
            os.makedirs(join(scene, 'sfm'))
        return car_poses

    # 生成car_poses.txt记录车身位姿
    def get_car_poses(self, scene_id):
        count = 1
        poses = self.get_ref_poses(scene_id)
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        destination_file_path = join(self.output_dir, scene_name, 'sfm', 'car_poses.txt')
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
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        destination_file_path = join(self.output_dir, scene_name, 'sfm', 'ref_cam_pose.txt')
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
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        destination_file_path = join(self.output_dir, scene_name, 'sfm', 'rig.json')
        camera_pose_txt_path = join(self.output_dir, scene_name, 'sfm', 'cam_pose.txt')
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
                        "image_prefix": camera_name+'/',
                        "rel_tvec": translation.tolist(),
                        "rel_qvec": [qw, qx, qy, qz]
                    }
                    json_data["cameras"].append(pose)
                json_data_list.append(json_data)
                json.dump(json_data_list, f, indent=4)

    # 生成cameras.txt记录相机参数
    def make_cameras_txt(self, scene_id):
        i = 0
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        txt_path = join(self.output_dir, scene_name, 'sfm', 'cameras.txt')
        with open(txt_path, 'w') as file:
            for camera_id, intrinsics_matrix in self.intrinsics[scene_id].items():
                print(intrinsics_matrix)
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

    def make_image_folder(self, scene_id):
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        destination_path = join(self.output_dir, scene_name, 'image')
        sample_token = scene_dict['first_sample_token']
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            line = str(sample['timestamp'])
            # 遍历该样本的所有 sample_data
            for cam_data_token in sample['data'].values():
                cam_data = self.nusc.get('sample_data', cam_data_token)
                if cam_data['sensor_modality'] == 'camera':
                    # 获取相机名称
                    cam_name = cam_data['channel']
                    cam_dir = os.path.join(destination_path, cam_name)
                    line = line + ' ' + cam_name + '/' + cam_data['filename'] + ' '
                    os.makedirs(cam_dir, exist_ok=True)
                    file_path = os.path.join(self.dataset_path, cam_data['filename'])
                    shutil.copy(file_path, cam_dir)
            sample_token = sample['next']
    
    
    
    # 生成images_list.txt,记录做重建所需的图片名称
    def make_images_list(self, scene_id):
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        destination_path = join(self.output_dir, scene_name, 'sfm', 'images_list.txt')
        root_folder = join(self.output_dir, scene_name)
        sfm_image_folder = join(root_folder, 'image')
        sfm_seg_folder = join(root_folder, 'seg')
        if not os.path.exists(sfm_seg_folder):
            os.makedirs(sfm_seg_folder)
        if not os.path.exists(sfm_image_folder):
            os.makedirs(sfm_image_folder)
        with open(destination_path, 'w') as output_file:
            for camera_name in self.camera_names:
                image_folder = os.path.join(sfm_image_folder,camera_name)
                image_names = sorted([f for f in listdir(image_folder) if isfile(join(image_folder, f))])
                for i, image_name in enumerate(image_names):
                    output_file.write(join(camera_name, image_name) + '\n')

    # 生成images.txt，记录每个图片的位姿
    def make_images_txt(self, scene_id):
        count = 1
        poses = self.get_ref_poses(scene_id)
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        destination_file_path = join(self.output_dir, scene_name, 'sfm', 'images.txt')
        list_path = join(self.output_dir, scene_name, "sfm", "images_list.txt")
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
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        folder_path = join(self.output_dir, scene_name, 'sfm')
        txt_path = join(folder_path, 'points3D.txt')
        with open(txt_path, 'w') as f:
            pass

    # 生成index.txt，以相机组的形式记录图像名称
    def make_index_txt(self, scene_id):
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        folder_path = join(self.output_dir, scene_name, 'sfm')
        destination_path = join(folder_path, 'index.txt')
        image_names = {}
        for camera_name in self.camera_names:
            image_folder = join(self.output_dir, scene_name, 'image', camera_name)
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
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        sfm_folder = join(self.output_dir, scene_name, 'sfm')
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
        self.make_image_folder(scene_id)
        self.make_images_list(scene_id)
        self.make_images_txt(scene_id)
        self.make_points3D_txt(scene_id)
        self.make_index_txt(scene_id)
        self.chouzhen(scene_id)

    # 完成整个工作流程
    def process_scene(self, scene_id):
        scene_dict = self.nusc.get('scene', scene_id)
        scene_name = scene_dict["name"]
        scene = join(self.output_dir, scene_name)
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
    parser.add_argument('--yaml_path', type=str, default='nuscene_sfm.yaml')
    args = parser.parse_args()
    mydataset = NeZhaDataset(args.yaml_path)
    mydataset.process_all()

