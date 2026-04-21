import os
import argparse
from tqdm import tqdm
import cv2
import json
import numpy as np
from multiprocessing import Pool
import shutil


def parse_intrinsic(data):
    content = next(iter(data.values()))
    param = content['param']
    intrinsic = dict()
    intrinsic["cam_K"] = np.array(param["cam_K"]["data"]).reshape(param["cam_K"]["rows"], param["cam_K"]["cols"])
    intrinsic["cam_dist"] = np.array(param["cam_dist"]["data"]).reshape(param["cam_dist"]["rows"],
                                                                        param["cam_dist"]["cols"])
    intrinsic["img_dist_w"] = param["img_dist_w"]
    intrinsic["img_dist_h"] = param["img_dist_h"]
    if "cam_K_undist" in param:
        intrinsic["cam_K_undist"] = np.array(param["cam_K_undist"]["data"]).reshape(param["cam_K_undist"]["rows"],
                                                                                    param["cam_K_undist"]["cols"])
    return intrinsic


def write_undistort_cam_K(intrinsics_root):
    for file in os.listdir(intrinsics_root):
        if file.endswith("intrinsic.json") and 'fisheye' not in file:
            intrinsics_path = os.path.join(intrinsics_root, file)
            print(intrinsics_path)
            with open(intrinsics_path, 'r') as f:
                data = json.load(f)
            intrinsic = parse_intrinsic(data)
            key = next(iter(data.keys()))
            image_size = (intrinsic["img_dist_w"], intrinsic["img_dist_h"])
            cam_dist = intrinsic["cam_dist"]
            old_K = intrinsic["cam_K"]
            if "cam_K_undist" in intrinsic:
                data[key]["param"].pop('cam_K_undist')
            # 根据cam_dist的维度判断是否为鱼眼相机
            if cam_dist.size == 4:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(old_K, cam_dist, image_size, np.eye(3))
            elif cam_dist.size == 8:
                new_K, _ = cv2.getOptimalNewCameraMatrix(old_K, cam_dist, image_size, 0, image_size)
            else:  # 未知相机类型
                raise ValueError("Unknown camera dist type")
            if 'front_wide' not in file:
                cam_K_undist = {"rows": new_K.shape[0], "cols": new_K.shape[1], "continuous": True,
                                "data": new_K.tolist()}
            else:
                cam_K_undist = {"rows": new_K.shape[0], "cols": new_K.shape[1], "continuous": True,
                                "data": old_K.tolist()}
            data[key]["param"]["cam_K_undist"] = cam_K_undist
            try:
                with open(intrinsics_path, 'w') as f:
                    json.dump(data, f, indent=4)
            except json.decoder.JSONDecodeError as e:
                print(f"JSON解码错误: {e}")
                continue


def get_cam_intrinsics(root_path, camera_intrinsic_map):
    intrinsics_map = dict()
    for camera_id, camera_name in camera_intrinsic_map.items():
        intrinsic_path = os.path.join(root_path, f"{camera_name}")
        if not os.path.exists(intrinsic_path):
            raise FileExistsError(f"{intrinsic_path} does not exist")
        with open(intrinsic_path, 'r') as f:
            data = json.load(f)
        intrinsic = parse_intrinsic(data)
        intrinsics_map[camera_id] = intrinsic
    return intrinsics_map


def undistort_image(input_path, output_path, old_K, new_K, cam_dist):
    image = cv2.imread(input_path)
    image_size = (image.shape[1], image.shape[0])
    # print(input_path)
    if cam_dist.size == 4:
        # print('cam_dist is 4')
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(old_K, cam_dist, np.eye(3), new_K, image_size, cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    elif cam_dist.size == 8:
        # print('cam_dist is 8')
        undistorted_img = cv2.undistort(image, old_K, cam_dist, None, new_K)
        # undistorted_img = cv2.undistort(image, old_K, cam_dist)
    else:  # 未知相机类型
        raise ValueError("Unknown camera dist type")
    cv2.imwrite(output_path, undistorted_img)


def get_imu_pose(folder_path):
    source_path = os.path.join(folder_path, 'global_imu_pose.txt')
    files_path = os.path.join(folder_path, 'files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    destination_path = os.path.join(files_path, 'global_imu_pose_INTER_aligned_time.txt')
    shutil.copy(source_path, files_path)
    with open(source_path, 'r') as f:
        lines = f.readlines()
    poses = {}
    for line in lines:
        parts = line.strip().split()
        poses[parts[0]] = parts[1:]
    label_file_path = os.path.join(folder_path, 'label_file')
    time_stamps = sorted(os.listdir(label_file_path))
    with open(destination_path, 'w') as file:
        for time_stamp in time_stamps:
            pose = poses[time_stamp]
            file.write(f'{time_stamp} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} \n')


def get_cam_pose_from_labelfile(folder_path, cam_type):
    source_path = os.path.join(folder_path, 'global_imu_pose.txt')
    files_path = os.path.join(folder_path, 'files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    destination_path = os.path.join(files_path, cam_type + '_global_imu_pose_INTER_aligned_time.txt')
    shutil.copy(source_path, files_path)
    with open(source_path, 'r') as f:
        lines = f.readlines()
    poses = []
    imu_time_stamps = []
    for line in lines:
        parts = line.strip().split()
        imu_time_stamps.append(float(parts[0]) / 1e9)
        poses.append(parts[1:])
    label_file_path = os.path.join(folder_path, 'label_file')
    time_stamps = sorted(os.listdir(label_file_path))
    cam_time_stamps = []
    cam_time_stamps_str = []
    for time_stamp in time_stamps:
        time_stamp_path = os.path.join(label_file_path, time_stamp)
        files = os.listdir(time_stamp_path)
        imgs = [file for file in files if
                file.endswith('.jpg') and 'undistort' not in file and 'resize' not in file and cam_type in file]
        img_name = imgs[0]
        print(img_name)
        last_dash_index = img_name.rfind("-")
        first_dot_index = img_name.rfind(".")
        substring = img_name[last_dash_index + 1:first_dot_index]
        cam_time_stamps.append(float(substring))
        cam_time_stamps_str.append(time_stamp)
    temp_index = 0
    with open(destination_path, 'w') as file:
        for j, cam_time_stamp in enumerate(cam_time_stamps):
            for i in range(temp_index, len(imu_time_stamps) - 1):
                if imu_time_stamps[i] < cam_time_stamp < imu_time_stamps[i + 1]:
                    pose = poses[i]
                    temp_index = i
                    file.write(
                        f'{cam_time_stamps_str[j]} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} \n')


def get_cam_pose_from_labelfile_added_time(folder_path, cam_type, added_time):
    source_path = os.path.join(folder_path, 'global_imu_pose.txt')
    files_path = os.path.join(folder_path, 'files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    destination_path = os.path.join(files_path,
                                    str(added_time) + '_' + cam_type + '_global_imu_pose_INTER_aligned_time.txt')
    shutil.copy(source_path, files_path)
    with open(source_path, 'r') as f:
        lines = f.readlines()
    poses = []
    imu_time_stamps = []
    for line in lines:
        parts = line.strip().split()
        imu_time_stamps.append(float(parts[0]) / 1e9)
        poses.append(parts[1:])
    label_file_path = os.path.join(folder_path, 'label_file')
    time_stamps = sorted(os.listdir(label_file_path))
    cam_time_stamps = []
    cam_time_stamps_str = []
    for time_stamp in time_stamps:
        time_stamp_path = os.path.join(label_file_path, time_stamp)
        files = os.listdir(time_stamp_path)
        imgs = [file for file in files if
                file.endswith('.jpg') and 'undistort' not in file and 'resize' not in file and cam_type in file]
        img_name = imgs[0]
        print(img_name)
        last_dash_index = img_name.rfind("-")
        first_dot_index = img_name.rfind(".")
        substring = img_name[last_dash_index + 1:first_dot_index]
        cam_time_stamps.append(float(substring)+added_time)
        cam_time_stamps_str.append(time_stamp)
    temp_index = 0
    with open(destination_path, 'w') as file:
        for j, cam_time_stamp in enumerate(cam_time_stamps):
            for i in range(temp_index, len(imu_time_stamps) - 1):
                if imu_time_stamps[i] < cam_time_stamp < imu_time_stamps[i + 1]:
                    pose = poses[i]
                    temp_index = i
                    file.write(
                        f'{cam_time_stamps_str[j]} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} \n')


def get_cam_pose_from_one_folder(folder_path, cam_type):
    source_path = os.path.join(folder_path, 'output', 'global_imu_pose.txt')
    files_path = os.path.join(folder_path, 'files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    destination_path = os.path.join(files_path, cam_type + '_global_imu_pose_INTER_aligned_time.txt')
    shutil.copy(source_path, files_path)
    with open(source_path, 'r') as f:
        lines = f.readlines()
    poses = []
    imu_time_stamps = []
    for line in lines:
        parts = line.strip().split()
        imu_time_stamps.append(float(parts[0]) / 1e9)
        poses.append(parts[1:])
    if os.path.exists(os.path.join(folder_path, 'image')):
        image_folder = os.path.join(folder_path, 'image', 'camera-' + cam_type)
    else:
        image_folder = os.path.join(folder_path, 'camera-' + cam_type)
    image_names = sorted(os.listdir(image_folder))
    cam_time_stamps = []
    cam_time_stamps_str = []
    for image_name in image_names:
        first_dot_index = image_name.rfind(".")
        substring = image_name[:first_dot_index]
        cam_time_stamps.append(float(substring))
        cam_time_stamps_str.append(substring)
    temp_index = 0
    with open(destination_path, 'w') as file:
        for j, cam_time_stamp in enumerate(cam_time_stamps):
            for i in range(temp_index, len(imu_time_stamps) - 1):
                if imu_time_stamps[i] < cam_time_stamp < imu_time_stamps[i + 1]:
                    pose = poses[i]
                    temp_index = i
                    file.write(
                        f'{cam_time_stamps_str[j]} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} \n')


def get_cam_pose_from_one_folder_added_time(folder_path, cam_type, added_time):
    source_path = os.path.join(folder_path, 'output', 'global_imu_pose.txt')
    files_path = os.path.join(folder_path, 'files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    destination_path = os.path.join(files_path,
                                    str(added_time) + '_' + cam_type + '_global_imu_pose_INTER_aligned_time.txt')
    shutil.copy(source_path, files_path)
    with open(source_path, 'r') as f:
        lines = f.readlines()
    poses = []
    imu_time_stamps = []
    for line in lines:
        parts = line.strip().split()
        imu_time_stamps.append(float(parts[0]) / 1e9)
        poses.append(parts[1:])
    if os.path.exists(os.path.join(folder_path, 'image')):
        image_folder = os.path.join(folder_path, 'image', 'camera-' + cam_type)
    else:
        image_folder = os.path.join(folder_path, 'camera-' + cam_type)
    image_names = sorted(os.listdir(image_folder))
    cam_time_stamps = []
    cam_time_stamps_str = []
    for image_name in image_names:
        first_dot_index = image_name.rfind(".")
        substring = image_name[:first_dot_index]
        cam_time_stamps.append(float(substring) + added_time)
        cam_time_stamps_str.append(substring)
    temp_index = 0
    with open(destination_path, 'w') as file:
        for j, cam_time_stamp in enumerate(cam_time_stamps):
            for i in range(temp_index, len(imu_time_stamps) - 1):
                if imu_time_stamps[i] < cam_time_stamp < imu_time_stamps[i + 1]:
                    pose = poses[i]
                    temp_index = i
                    file.write(
                        f'{cam_time_stamps_str[j]} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} \n')


def undistort_from_labelfile(folder_path):
    camera_name_map = {"71": "front_narrow-intrinsic.json", "73": "front_wide-intrinsic.json",
                       "75": "back_left-intrinsic.json", "76": "front_left-intrinsic.json",
                       "77": "back_right-intrinsic.json", "78": "front_right-intrinsic.json",
                       "83": "back-intrinsic.json"}
    intrinsic_root = os.path.join(folder_path, 'param')
    # 原地覆盖内参文件
    write_undistort_cam_K(intrinsic_root)
    intrinsics_map = get_cam_intrinsics(intrinsic_root, camera_name_map)
    label_file_path = os.path.join(folder_path, 'label_file')
    time_stamps = sorted(os.listdir(label_file_path))
    input_paths = []
    output_paths = []
    old_Ks = []
    new_Ks = []
    cam_dists = []
    for time_stamp in time_stamps:
        time_stamp_path = os.path.join(label_file_path, time_stamp)
        files = os.listdir(time_stamp_path)
        imgs = [file for file in files if file.endswith('.jpg')]
        for img in imgs:
            if "undistort" in img:
                # print(os.path.join(time_stamp_path, img))
                # os.remove(os.path.join(time_stamp_path, img))
                continue
            for camera_id, intrinsic_name in camera_name_map.items():
                if 'camera-' + camera_id + '-' in img and "undistort" not in img and 'resize' not in img:
                    sub_dir = os.path.join(time_stamp_path, img)
                    new_out_dir = os.path.join(time_stamp_path, 'undistort-' + img)
                    input_paths.append(sub_dir)
                    output_paths.append(new_out_dir)
                    old_Ks.append(intrinsics_map[camera_id]["cam_K"])
                    new_Ks.append(intrinsics_map[camera_id]["cam_K_undist"])
                    cam_dists.append(intrinsics_map[camera_id]["cam_dist"])

    # 多进程异步进行
    with Pool(args.jobs) as p:
        for input_path, output_path, old_K, new_K, cam_dist in zip(input_paths, output_paths, old_Ks, new_Ks,
                                                                   cam_dists):
            p.apply_async(undistort_image, (input_path, output_path, old_K, new_K, cam_dist))
        p.close()
        p.join()


def undistort_from_one_folder(folder_path):
    camera_name_map = {"1": "front_narrow-intrinsic.json", "2": "back-intrinsic.json",
                       "5": "front_left-intrinsic.json", "4": "back_left-intrinsic.json",
                       "6": "front_right-intrinsic.json", "7": "back_right-intrinsic.json",
                       "0": "front_wide-intrinsic.json"
                       }
    # intrinsic_root = '/dataset/rtfbag/EP40-38/SensorCalibration/42_new_params'
    intrinsic_root = os.path.join(folder_path, 'param')
    # 原地覆盖内参文件
    write_undistort_cam_K(intrinsic_root)
    intrinsics_map = get_cam_intrinsics(intrinsic_root, camera_name_map)
    input_paths = []
    output_paths = []
    old_Ks = []
    new_Ks = []
    cam_dists = []
    for cam_type in camera_name_map.keys():
        if os.path.exists(os.path.join(folder_path, 'image')):
            image_folder = os.path.join(folder_path, 'image', 'camera-' + cam_type)
            undistort_image_folder = os.path.join(folder_path, 'image', 'camera-' + cam_type + '-undistort')
        else:
            image_folder = os.path.join(folder_path, 'camera-' + cam_type)
            undistort_image_folder = os.path.join(folder_path, 'camera-' + cam_type + '-undistort')
        if not os.path.exists(undistort_image_folder):
            os.mkdir(undistort_image_folder)
        image_names = sorted(os.listdir(image_folder))
        for image_name in image_names:
            sub_dir = os.path.join(image_folder, image_name)
            new_out_dir = os.path.join(undistort_image_folder, image_name)
            input_paths.append(sub_dir)
            output_paths.append(new_out_dir)
            old_Ks.append(intrinsics_map[cam_type]["cam_K"])
            new_Ks.append(intrinsics_map[cam_type]["cam_K_undist"])
            cam_dists.append(intrinsics_map[cam_type]["cam_dist"])

    # 多进程异步进行
    with Pool(args.jobs) as p:
        for input_path, output_path, old_K, new_K, cam_dist in zip(input_paths, output_paths, old_Ks, new_Ks,
                                                                   cam_dists):
            p.apply_async(undistort_image, (input_path, output_path, old_K, new_K, cam_dist))
        p.close()
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, help="7相机图片文件夹路径")
    parser.add_argument("-j", "--jobs", type=int, default=32, help="多进程数")
    args = parser.parse_args()
    '''root = '/vepfs_dataset/wwh/orin_2'
    # root = args.root
    if not os.path.exists(root):
        raise FileExistsError(f"{root} not exists")
    sub_folder_paths = sorted(os.listdir(root))
    for sub_folder_path in sub_folder_paths:
        print(sub_folder_path)
        sub_sub_folder_paths = sorted(os.listdir(os.path.join(root, sub_folder_path)))
        for sub_sub_folder_path in sub_sub_folder_paths:
            data_folder = os.path.join(root, sub_folder_path, sub_sub_folder_path)
            print(data_folder)
            try:
                get_imu_pose(data_folder)
                undistort_from_labelfile(data_folder)
            except Exception as e:
                print(f"An error occurred: {e}. Moving on to the next sub_folder.")
                continue'''
    root = '/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/data/EP41-ORIN-0S14-00G_20240523_081429/EP41-ORIN-0S14-00G_FULL_MCAP_02.22.03_20240414-105525_52.mcap/params'
    write_undistort_cam_K(root)
    '''root = '/vepfs_dataset/handsome_man_with_handsome_data/orin13/EP41-ORIN-0S13-00G_FULL_MCAP_02.22.03_20240410-173615_51.mcap'
    #get_cam_pose(root, '71')
    #get_cam_pose(root, '73')
    get_cam_pose_from_labelfile_added_time(root, '73', 0.03)
    get_cam_pose_from_labelfile_added_time(root, '71', 0.03)
    get_cam_pose_from_labelfile_added_time(root, '73', 0.05)
    get_cam_pose_from_labelfile_added_time(root, '71', 0.05)
    get_cam_pose_from_labelfile_added_time(root, '73', 0.08)
    get_cam_pose_from_labelfile_added_time(root, '71', 0.08)
    #undistort_from_one_folder(root)
    #undistort_from_labelfile(root)'''
    '''get_imu_pose(root)
    undistort_from_labelfile(root)'''
