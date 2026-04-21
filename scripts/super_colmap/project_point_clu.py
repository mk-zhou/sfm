import cv2
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
import random

def read_camera_data(file_path):
    camera_data = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            camera_id, model, width, height, *params = line.split()
            camera_data[int(camera_id)] = {
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": [float(param) for param in params]
            }
    return camera_data


def read_image_data(file_path):
    image_data = {}
    with open(file_path, "r") as f:
        for line1, line2 in zip(f, f):
            if line1.startswith("#") or line1.strip() == "":
                continue
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line1.split()

            # 解析第二行数据，并按每三个数据为一组进行处理
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


def read_track_data(file_path, image_data, camera_data, output_file):
    points_by_pixel_value = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            point3d_id, x, y, z, r, g, b, error, *track_data = line.split()
            track = [tuple(map(int, track_data[i:i + 2])) for i in range(0, len(track_data), 2)]

            pixel_values = []  # 保存每个像素值
            for image_id, point2d_idx in track:
                image = image_data.get(image_id)
                if image:
                    camera = camera_data.get(image["camera_id"])
                    if camera:
                        image_width, image_height = camera["width"], camera["height"]
                        if 0 <= point2d_idx < len(image["points2d"]):
                            point2d = image["points2d"][point2d_idx]
                            proj_x = point2d[0]
                            proj_y = point2d[1]
                            # print(f"Point3D ID: {point3d_id}, Image ID: {image_id}, Projection Point ID: {point2d_idx}")
                            # print(f"Projection X: {proj_x}, Projection Y: {proj_y}")

                            # 读取对应的语义灰度图像
                            image_name = image["name"]  # 图片名称
                            sem_image_path = f"sem_images/{image_name[:-4]}._bin.png"  # 生成对应的语义灰度图像路径
                            sem_image = cv2.imread(sem_image_path, cv2.IMREAD_GRAYSCALE)
                            if sem_image is not None:
                                pixel_value = sem_image[int(proj_y), int(proj_x)]
                                pixel_values.append(pixel_value)
                                # print(f"Pixel Value: {pixel_value}")
                            else:
                                print("Failed to read semantic image")

                        else:
                            print(f"Invalid Point2D ID: {point2d_idx} in Image ID: {image_id}")
                    else:
                        print(f"Camera ID: {image['camera_id']} not found for Image ID: {image_id}")
                else:
                    print(f"Image ID: {image_id} not found")

            if pixel_values:
                # 投票选出出现次数最多的像素值
                voted_pixel_value = Counter(pixel_values).most_common(1)[0][0]
                if voted_pixel_value not in [0, 1, 19, 20, 21, 22, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,]:
                    print(f"Voted Pixel Value (Point ID: {point3d_id}): {voted_pixel_value}\n")

                    r, g, b = get_rgb_from_pixel_value(voted_pixel_value)

                    # 写入修改后的数据到sem_points3D.txt文件中
                    output_file.write(f"{point3d_id} {x} {y} {z} {r} {g} {b} {error} {' '.join(track_data)}\n")
                else:
                    if voted_pixel_value not in points_by_pixel_value:
                        points_by_pixel_value[voted_pixel_value] = []
                    points_by_pixel_value[voted_pixel_value].append((point3d_id, x, y, z, r, g, b, error, *track_data))
            else:
                print(f"No pixel values found for Point3D ID: {point3d_id}\n")
    return points_by_pixel_value


def cluster_points_by_pixel(points_by_pixel_value, eps, min_samples):
    clustered_points = {}
    cluster_colors = {}
    used_colors = set()  # 记录已使用的颜色
    for pixel_value, points in points_by_pixel_value.items():

        coordinates = [(float(x), float(y), float(z)) for _, x, y, z, *_ in points]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coordinates)
        print("labels", labels)
        for i, point in enumerate(points):
            label = labels[i]
            if label == -1:
                rgb = (255, 255, 255)  # 未成功聚类的点设置为白色
                if -1 not in clustered_points:
                    clustered_points[-1] = []
            else:
                label = pixel_value * 100 + label
                if label not in clustered_points:
                    rgb = (255, 255, 255)
                    clustered_points[label] = []
                    # 生成新的随机颜色，直到找到一个未使用且符合条件的颜色
                    while rgb in used_colors or rgb == (255, 255, 255) or rgb == (0, 0, 0) or rgb == (0, 255, 0) or rgb is None:
                        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    rgb = cluster_colors.get(label)
            # print("label:", label)
            # print("rgb:", rgb)
            clustered_points[label].append(point)
            cluster_colors[label] = rgb
            used_colors.add(rgb)
    for label, points in clustered_points.items():
        rgb = cluster_colors[label]
        new_points = []
        for point in points:
            point3d_id, x, y, z, r, g, b, error, *track_data = point
            new_points.append((point3d_id, x, y, z, rgb[0], rgb[1], rgb[2], error, *track_data))

        clustered_points[label].clear()
        clustered_points[label].extend(new_points)

    return clustered_points, cluster_colors


def write_clustered_points_to_file(clustered_points, cluster_colors, output_file):
    for label, points in clustered_points.items():
        rgb = cluster_colors[label]  # 获取该簇对应的 RGB 值
        for point in points:
            point3d_id, x, y, z, _, _, _, error, *track_data = point
            output_file.write(f"{point3d_id} {x} {y} {z} {rgb[0]} {rgb[1]} {rgb[2]} {error} {' '.join(track_data)}\n")


def get_rgb_from_pixel_value(pixel_value):
    ###红色 [255, 0, 0]房屋及其他；绿色[0, 255, 0]地面；蓝色 [0, 0, 255]动态物
    rgb_mapping = {
        0: [0, 0, 255],
        1: [0, 0, 255],
        2: [0, 0, 0],
        3: [0, 0, 0],
        4: [0, 0, 0],
        5: [0, 0, 0],
        6: [0, 0, 0],
        7: [0, 255, 0],
        8: [0, 255, 0],
        9: [0, 255, 0],
        10: [0, 255, 0],
        11: [0, 255, 0],
        12: [0, 255, 0],
        13: [0, 255, 0],
        14: [0, 255, 0],
        15: [0, 255, 0],
        16: [0, 255, 0],
        17: [0, 0, 0],
        18: [0, 0, 0],
        19: [0, 0, 255],
        20: [0, 0, 255],
        21: [0, 0, 255],
        22: [0, 0, 255],
        23: [0, 255, 0],
        24: [0, 255, 0],
        25: [0, 0, 0],
        26: [0, 0, 0],
        27: [0, 0, 0],
        28: [0, 0, 0],
        29: [0, 0, 0],
        30: [0, 0, 0],
        31: [0, 0, 0],
        32: [0, 0, 0],
        33: [0, 0, 0],
        34: [0, 0, 0],
        35: [0, 0, 0],
        36: [0, 0, 0],
        37: [0, 0, 0],
        38: [0, 0, 0],
        39: [0, 0, 0],
        40: [0, 0, 0],
        41: [0, 0, 0],
        42: [0, 0, 0],
        43: [0, 0, 0],
        44: [0, 0, 0],
        45: [0, 0, 0],
        46: [0, 0, 0],
        47: [0, 0, 0],
        48: [0, 0, 0],
        49: [0, 0, 0],
        50: [0, 0, 0],
        51: [0, 0, 0],
        52: [0, 0, 255],
        53: [0, 0, 255],
        54: [0, 0, 255],
        55: [0, 0, 255],
        56: [0, 0, 255],
        57: [0, 0, 255],
        58: [0, 0, 255],
        59: [0, 0, 255],
        60: [0, 0, 255],
        61: [0, 0, 255],
        62: [0, 0, 255],
        63: [0, 0, 255],
        64: [0, 0, 255],
        65: [0, 0, 255],
    }
    return rgb_mapping.get(pixel_value, [0, 0, 0])



def main():
    camera_data = read_camera_data("model/cameras.txt")
    image_data = read_image_data("model/images.txt")

    # 创建 sem_points3D.txt 文件并打开以进行写操作
    output_file = open("sem_points3D.txt", "w")

    # 调用 read_track_data 函数并传入 output_file 参数
    points_by_pixel_value = read_track_data("model/points3D.txt", image_data, camera_data, output_file)
    print("begin to cluster")

    eps = 0.5
    min_samples = 5
    clustered_points, cluster_colors = cluster_points_by_pixel(points_by_pixel_value, eps, min_samples)
    write_clustered_points_to_file(clustered_points, cluster_colors, output_file)
    # 关闭文件
    output_file.close()


if __name__ == "__main__":
    main()
