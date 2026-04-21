import open3d as o3d
import numpy as np
import cv2
# from transforms3d import euler, quaternions
import os
from os.path import join
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil


def get_sfm_img_path(project_dir, type='pgo_'):
    dense_ply_dir = ''
    poses = []
    img_paths = []
    names = []
    for sub_folder in os.listdir(project_dir):
        if type + 'sfm' in sub_folder and os.path.isdir(os.path.join(project_dir, sub_folder)):
            chouzhen_dir = os.path.join(project_dir, sub_folder, 'chouzhen')
            for sub_sub_folder in os.listdir(chouzhen_dir):
                if 'mapper' in sub_sub_folder and os.path.isdir(os.path.join(chouzhen_dir, sub_sub_folder)):
                    txt_dir = os.path.join(chouzhen_dir, sub_sub_folder, 'txt')
                    # dense_ply_dir = os.path.join(chouzhen_dir, 'dense', 'dense_road_rm_uniZ.ply')
                    dense_ply_dir = os.path.join(chouzhen_dir, 'dense', 'dense_rmdy_fil.ply')
                    if os.path.exists(txt_dir) and os.path.exists(dense_ply_dir):
                        image_txt = os.path.join(txt_dir, 'images.txt')
                        print(image_txt)
                        with  open(image_txt, 'r') as f:
                            lines = f.readlines()[4::2]
                            for line in lines:
                                parts = line.strip().split(' ')
                                if int(parts[0]) % 10 != 1 or parts[-2] != '1':
                                    continue
                                img_path = os.path.join(project_dir, 'rawCamera', parts[-1])
                                name = parts[-1].split('/')[-1]
                                names.append(name)
                                pose = np.eye(4)
                                quat = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[1])]
                                rqut = Rotation.from_quat(quat)
                                rotation = rqut.as_matrix()
                                translation = parts[5:8]
                                pose[:3, :3] = rotation
                                pose[:3, 3] = translation
                                print(pose)
                                print(img_path)
                                print(name)
                                poses.append(pose)
                                img_paths.append(img_path)
    print(dense_ply_dir)
    return dense_ply_dir, poses, img_paths, names


def get_slam_img_path(project_dir):
    #dense_ply_dir = join(project_dir, 'filtered_global_map.pcd')
    dense_ply_dir = join(project_dir, 'global_map.pcd')
    poses = []
    img_paths = []
    names = []
    chouzhen_dir = os.path.join(project_dir, 'sfm')
    image_txt = os.path.join(chouzhen_dir, 'slam_images.txt')
    print(image_txt)
    with  open(image_txt, 'r') as f:
        lines = f.readlines()[::2]
        for line in lines:
            parts = line.strip().split(' ')
            if int(parts[0]) % 40 != 1 or parts[-2] != '1':
                continue
            img_path = os.path.join(project_dir, 'rawCamera', parts[-1])
            name = parts[-1].split('/')[-1]
            names.append(name)
            pose = np.eye(4)
            quat = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[1])]
            rqut = Rotation.from_quat(quat)
            rotation = rqut.as_matrix()
            translation = parts[5:8]
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            print(pose)
            print(img_path)
            print(name)
            poses.append(pose)
            img_paths.append(img_path)
    print(dense_ply_dir)
    return dense_ply_dir, poses, img_paths, names


def get_sfm_img_path_clip(project_dir, clip_id, type='pgo_'):
    dense_ply_dir = ''
    poses = []
    img_paths = []
    names = []
    for sub_folder in os.listdir(project_dir):
        if type + 'sfm' in sub_folder and os.path.isdir(os.path.join(project_dir, sub_folder)):
            chouzhen_dir = os.path.join(project_dir, sub_folder, 'chouzhen')
            for sub_sub_folder in os.listdir(chouzhen_dir):
                if 'clips' in sub_sub_folder and os.path.isdir(os.path.join(chouzhen_dir, sub_sub_folder)):
                    clip_path = os.path.join(chouzhen_dir, sub_sub_folder, str(clip_id))
                    txt_dir = os.path.join(clip_path, 'rig_mapper_txt')
                    dense_ply_dir = os.path.join(clip_path, 'dense', 'dense_rmdy_fil_0.3.ply')
                    if os.path.exists(txt_dir) and os.path.exists(dense_ply_dir):
                        image_txt = os.path.join(txt_dir, 'images.txt')
                        with  open(image_txt, 'r') as f:
                            lines = f.readlines()[4::2]
                            for line in lines:
                                parts = line.strip().split(' ')
                                if int(parts[0]) % 3 != 1 or parts[-2] != '1':
                                    # if  parts[-2] != '1':
                                    continue
                                img_path = os.path.join(project_dir, 'rawCamera', parts[-1])
                                name = parts[-1].split('/')[-1]
                                names.append(name)
                                pose = np.eye(4)
                                quat = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[1])]
                                rqut = Rotation.from_quat(quat)
                                rotation = rqut.as_matrix()
                                translation = parts[5:8]
                                pose[:3, :3] = rotation
                                pose[:3, 3] = translation
                                print(pose)
                                print(img_path)
                                print(name)
                                poses.append(pose)
                                img_paths.append(img_path)
    print(dense_ply_dir)
    return dense_ply_dir, poses, img_paths, names


def image_to_point_cloud(image):
    width, height = image.size
    pixels = np.array(image)

    # Create pixel grid
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten coordinates and set Z coordinate to 0
    points = np.column_stack((xx.flatten(), yy.flatten(), np.zeros(width * height)))

    # Flatten color pixels
    colors = pixels.reshape((-1, 3)) / 255.0

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # print(colors)
    o3d.io.write_point_cloud("or.ply", point_cloud)
    return point_cloud


def make_video(folder, name):
    image_files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    first_image_path = os.path.join(folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    output_video_path = join(folder, name + '.avi')
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 3, (width, height))

    # 将图片逐帧写入视频
    for image_file in tqdm(image_files):
        image_path = os.path.join(folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    # 释放资源
    video_writer.release()


def merge_img(folder, name, clip_count):
    input_img_base = os.path.join(folder, 'clip_' + name)
    output_img_folder = os.path.join(folder, 'merge_' + name)
    for i in range(clip_count):
        input_img_path = input_img_base + str(i)
        for file in os.listdir(input_img_path):
            print(os.path.join(input_img_path, file))
            shutil.copy(os.path.join(input_img_path, file), output_img_folder)


def project_points(project_dir, type, intrinsic_matrix, slam, clip_id=-1):
    # Convert pose (translation + quaternion) to transformation matrix
    if not slam:
        if clip_id != -1:
            ply_dir, poses, img_paths, names = get_sfm_img_path_clip(project_dir, clip_id, type)
            re_output_dir = join(project_dir, type + 'sfm', 'chouzhen', 'clip_reproject' + str(clip_id))
            de_output_dir = join(project_dir, type + 'sfm', 'chouzhen', 'clip_depth' + str(clip_id))
        else:
            ply_dir, poses, img_paths, names = get_sfm_img_path(project_dir, type)
            re_output_dir = join(project_dir, type + 'sfm', 'chouzhen', 'reproject')
            de_output_dir = join(project_dir, type + 'sfm', 'chouzhen', 'depth')
        if not os.path.exists(re_output_dir):
            os.makedirs(re_output_dir)
    else:
        ply_dir, poses, img_paths, names = get_slam_img_path(project_dir)
        de_output_dir = join(project_dir, 'depth')
    projected_points = []

    if not os.path.exists(de_output_dir):
        os.makedirs(de_output_dir)
    if not slam:
        for i, (pose, img_path, name) in enumerate(zip(poses, img_paths, names)):
            point_cloud = o3d.io.read_point_cloud(ply_dir)
            points = np.asarray(point_cloud.points)
            colors = (np.asarray(point_cloud.colors) * 255).astype(int)
            # 将三维点添加齐次坐标
            homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
            # 将三维点从世界坐标系转换为相机坐标系
            points_camera_coords = np.dot(pose, homogeneous_points.T).T[:, :3]
            # 通过z是否大于零
            distances = np.linalg.norm(points_camera_coords, axis=1)
            points_camera_coords_after_mask = points_camera_coords[
                np.logical_and(0 < points_camera_coords[:, 2], points_camera_coords[:, 2] < 30)]
            colors_after_mask = colors[np.logical_and(0 < points_camera_coords[:, 2], points_camera_coords[:, 2] < 30)]

            distances_after_mask = distances[
                np.logical_and(0 < points_camera_coords[:, 2], points_camera_coords[:, 2] < 30)]
            point_2d = np.dot(intrinsic_matrix, points_camera_coords_after_mask.T).T
            # print(point_2d)
            # 对每一个点除以第3个元素
            third_column = point_2d[:, 2][:, np.newaxis]
            # print(third_column)
            point_2d = (point_2d / third_column).astype(int)
            # point_2d = np.round(point_2d[:2]).astype(int)
            # print(point_2d)
            # 判断是否在图片内
            point_2d_in_img = point_2d[
                (point_2d[:, 0] >= 0) & (point_2d[:, 0] < 1920) & (point_2d[:, 1] >= 0) & (point_2d[:, 1] < 1080)]
            colors_in_img = colors_after_mask[
                (point_2d[:, 0] >= 0) & (point_2d[:, 0] < 1920) & (point_2d[:, 1] >= 0) & (
                        point_2d[:, 1] < 1080)]
            distances_in_img = distances_after_mask[
                (point_2d[:, 0] >= 0) & (point_2d[:, 0] < 1920) & (point_2d[:, 1] >= 0) & (
                        point_2d[:, 1] < 1080)]
            # 把n*3的颜色矩阵，每一个元素分别是RGB，n*3的位姿矩阵，前两位是x,y坐标，第三位是1
            # 先变换成n*5的array，再把这个array变换成h*w*3的图片array，其中n是有效点的个数，h是图片的高1080，w是图片的宽1920
            # 然后把图片array保存成图片
            # Create an image array
            re_image_array = np.zeros((1080, 1920, 3), dtype=np.uint8)
            re_image_array.fill(255)
            for point, color in zip(point_2d_in_img, colors_in_img):
                re_image_array[point[1], point[0]] = color
            Image.fromarray(re_image_array).save(join(re_output_dir, name))
            print("Saved image: ", join(re_output_dir, name))
            de_image_array = np.zeros((1080, 1920), dtype=np.float64)
            de_image_array.fill(255)
            # print(np.max(distances_in_img))
            for point, distance in zip(point_2d_in_img, distances_in_img):
                de_image_array[point[1], point[0]] = distance
            np.save(join(de_output_dir, name.replace('.jpg', '.npy')), de_image_array)
            Image.fromarray(de_image_array.astype(np.uint8)).save(join(de_output_dir, name))
            print("Saved image: ", join(de_output_dir, name))
        make_video(re_output_dir, 're')
        make_video(de_output_dir, 'de')
    else:
        for i, (pose, img_path, name) in enumerate(zip(poses, img_paths, names)):
            point_cloud = o3d.io.read_point_cloud(ply_dir)
            points = np.asarray(point_cloud.points)
            # 将三维点添加齐次坐标
            homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
            # 将三维点从世界坐标系转换为相机坐标系
            points_camera_coords = np.dot(pose, homogeneous_points.T).T[:, :3]
            # 通过z是否大于零
            distances = np.linalg.norm(points_camera_coords, axis=1)
            points_camera_coords_after_mask = points_camera_coords[
                np.logical_and(0 < points_camera_coords[:, 2], points_camera_coords[:, 2] < 30)]
            distances_after_mask = distances[
                np.logical_and(0 < points_camera_coords[:, 2], points_camera_coords[:, 2] < 30)]
            point_2d = np.dot(intrinsic_matrix, points_camera_coords_after_mask.T).T
            # print(point_2d)
            # 对每一个点除以第3个元素
            third_column = point_2d[:, 2][:, np.newaxis]
            point_2d = (point_2d / third_column).astype(int)
            # 判断是否在图片内
            point_2d_in_img = point_2d[
                (point_2d[:, 0] >= 0) & (point_2d[:, 0] < 1920) & (point_2d[:, 1] >= 0) & (point_2d[:, 1] < 1080)]
            distances_in_img = distances_after_mask[
                (point_2d[:, 0] >= 0) & (point_2d[:, 0] < 1920) & (point_2d[:, 1] >= 0) & (
                        point_2d[:, 1] < 1080)]
            # 把n*3的颜色矩阵，每一个元素分别是RGB，n*3的位姿矩阵，前两位是x,y坐标，第三位是1
            # 先变换成n*5的array，再把这个array变换成h*w*3的图片array，其中n是有效点的个数，h是图片的高1080，w是图片的宽1920
            # 然后把图片array保存成图片
            # Create an image array
            de_image_array = np.zeros((1080, 1920), dtype=np.float64)
            de_image_array.fill(255)
            for point, distance in zip(point_2d_in_img, distances_in_img):
                de_image_array[point[1], point[0]] = distance
            # print(de_image_array)
            np.save(join(de_output_dir, name.replace('.jpg', '.npy')), de_image_array)
            Image.fromarray(de_image_array.astype(np.uint8)).save(join(de_output_dir, name))
            print("Saved image: ", join(de_output_dir, name))
        make_video(de_output_dir, 'de')
    return projected_points


if __name__ == '__main__':
    intrinsic_matrix = np.array([[771.0563930202114, 0, 986.6638539339681],
                                 [0, 771.1950781923796, 512.6614906278281],
                                 [0, 0, 1]])
    '''project_points(
        '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-14-22-03_110.bag', '',
        intrinsic_matrix, 0)
    project_points(
        '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-14-22-03_110.bag', '',
        intrinsic_matrix, 1)'''
    # make_video('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-14-22-03_110.bag/sfm/chouzhen/clip_depth0','de')
    # merge_img('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-14-22-03_110.bag/sfm/chouzhen','depth',2)
    # make_video('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-14-22-03_110.bag/sfm/chouzhen/merge_depth','de')
    project_points('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag',
                   '', intrinsic_matrix, False)
