import os
import cv2
import numpy as np
import open3d as o3d
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from os.path import join


# 输入：images.txt的路径
# 输出：记录每张图片的信息的字典
def read_image_data(file_path):
    image_data = {}
    with open(file_path, "r") as f:
        for line1, line2 in zip(f, f):
            if line1.startswith("#") or line1.strip() == "":
                continue
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line1.split()

            points2d = []
            point_data = line2.split(";")[0].split()
            for i in range(0, len(point_data), 3):
                x, y, point3d_id = map(float, point_data[i:i + 3])
                points2d.append((x, y, point3d_id))

            image_data[int(image_id)] = {
                "qw": float(qw),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "tx": float(tx),
                "ty": float(ty),
                "tz": float(tz),
                "camera_id": int(camera_id),
                "name": name,
                "points2d": points2d
            }
    return image_data


# 输入：三维点的信息，图片信息，语义灰度图像母文件夹的路径
def process_point(point, image_data, seg_folder):
    pixel_values = []  # 保存每个像素值
    track = point['track']
    for image_id, point2d_idx in track:
        image = image_data.get(image_id)
        if 0 <= point2d_idx < len(image["points2d"]):
            point2d = image["points2d"][point2d_idx]
            proj_x = point2d[0]
            proj_y = point2d[1]

            # 读取对应的语义灰度图像
            image_name = image["name"]  # 图片名称
            seg_name = image_name.replace(".jpg", ".jpg.png")
            sem_image_path = os.path.join(seg_folder, seg_name)  # 生成对应的语义灰度图像路径
            sem_image = cv2.imread(sem_image_path, cv2.IMREAD_GRAYSCALE)
            if sem_image is not None:
                pixel_value = sem_image[int(proj_y), int(proj_x)]
                pixel_values.append(pixel_value)
            else:
                print("Failed to read semantic image")
        else:
            print(f"Invalid Point2D ID: {point2d_idx} in Image ID: {image_id}")

    if pixel_values:
        # 投票选出出现次数最多的像素值
        voted_pixel_value = Counter(pixel_values).most_common(1)[0][0]

        if voted_pixel_value != 0:  # 如果像素值不等于0，则保存
            return (
                [float(point['x']), float(point['y']), float(point['z'])],
                [float(point['r']) / 255, float(point['g']) / 255, float(point['b']) / 255]
            )
    return None


# 输入：points3D.txt的路径，图片信息，语义灰度图像母文件夹的路径，输出点云文件的路径
# 得到地面点云
def process_point_data(file_path, image_data, seg_folder, output_file):
    non_zero_gray_points = []  # 保存灰度值不为0的点的坐标和颜色信息
    num_points = 0  # 记录已遍历的点的数量
    colors = []
    with open(file_path, "r") as f:
        points = []
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            point3d_id, x, y, z, r, g, b, error, *track_data = line.split()
            track = [tuple(map(int, track_data[i:i + 2])) for i in range(0, len(track_data), 2)]

            points.append({
                "point3d_id": int(point3d_id),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "r": int(r),
                "g": int(g),
                "b": int(b),
                "error": float(error),
                "track": track
            })

            num_points += 1

    print(f"Total number of points: {num_points}")

    with ThreadPoolExecutor() as executor, tqdm(total=len(points)) as pbar:
        # 多线程处理每个点
        for result in executor.map(process_point, points, [image_data] * len(points), [seg_folder] * len(points)):
            pbar.update(1)
            if result is not None:
                non_zero_gray_points.append(result[0])
                colors.append(result[1])

    if non_zero_gray_points:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(non_zero_gray_points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        o3d.io.write_point_cloud(output_file, point_cloud)
        print(f"Point cloud saved to {output_file}")
    else:
        print("No non-zero gray points found")


def get_road_ply(scene):
    # 文件路径
    chouzhen_path = join(scene, 'sfm', 'chouzhen')
    points3D_txt = join(chouzhen_path, 'rig_mapper', 'txt', 'points3D.txt')
    image_txt = join(chouzhen_path, 'rig_mapper', 'txt', 'images.txt')
    seg_folder = join(chouzhen_path, 'seg_road')
    output_pcd_file = join(chouzhen_path, 'rig_mapper', 'road.ply')

    # 读取数据
    image_data = read_image_data(image_txt)

    # 处理点数据
    process_point_data(points3D_txt, image_data, seg_folder, output_pcd_file)
    print("save end")


if __name__ == '__main__':
    get_road_ply('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83')
