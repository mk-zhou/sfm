import open3d as o3d
import numpy as np
import os
from scipy.spatial import cKDTree
from tqdm import tqdm
from os.path import join, isfile, exists


# 输入：雷达点云路径，sfm点云路径，阈值（当sfm点附近多少米没有雷达点时，滤掉这个点）
# 输出：平均距离误差，滤除点数量
def compute_distance(point_cloud_lidar, point_cloud_sfm, threshold):
    # 读取点云文件
    pcd_lidar = o3d.io.read_point_cloud(point_cloud_lidar)
    pcd_sfm = o3d.io.read_point_cloud(point_cloud_sfm)

    # 获取点云坐标
    points_lidar = np.asarray(pcd_lidar.points)
    points_sfm = np.asarray(pcd_sfm.points)

    # 打印点云数量
    num_points = len(points_sfm)
    print("sfm点云数量: ", num_points)

    # 构建k-d树
    kdtree = cKDTree(points_lidar)

    # 计算每个点在point_cloud_sfm中距离最近的点的距离
    distances = []
    continue_count = 0  # 初始化计数变量
    for i, point_sfm in enumerate(tqdm(points_sfm, desc="Calculate_distance")):
        _, idx = kdtree.query(point_sfm)  # 寻找最近邻点
        nearest_point = points_lidar[idx]

        distance = np.linalg.norm(point_sfm - nearest_point)  # 计算距离
        if distance > threshold:
            continue_count += 1  # 若距离大于阈值，则计数加一并跳过当前点
            continue  # 跳过当前点

        z_error = abs(point_sfm[2] - nearest_point[2])  # 计算z轴误差
        distances.append(z_error)  # 将z轴误差加入总距离

    # 计算平均距离误差
    average_distance_error = np.mean(distances)
    print("滤除点数量: ", continue_count)

    # 打印结果
    print("平均距离误差: ", average_distance_error)
    return average_distance_error, continue_count


# 输入：场景路径
# 输出：车辆位置
def get_car_positions(scene):
    odometry_path = join(scene, "sfm", 'chouzhen', 'rig_mapper', 'new_car_poses.txt')
    if not isfile(odometry_path):
        return None
    # 每10个时刻取一个
    with open(odometry_path, 'r') as f:
        lines = f.readlines()[::20]
    car_positions = {}
    for line in lines:
        parts = line.strip().split(' ')
        name = parts[-1]
        # 查找最后一个斜杠和最后一个点的索引
        slash_index = name.rfind("/")
        dot_index = name.rfind(".")

        # 提取子字符串
        timestamp = name[slash_index + 1:dot_index]
        t = np.array(parts[5:8], dtype=np.float64)
        car_positions[timestamp] = t
    return car_positions


# 输入：场景路径，车辆位置，半径，索引（第几个车），类型
def get_local_ply(scene, position, radius, index, type):
    if type == 'sfm':
        ply_path = join(scene, "sfm", 'chouzhen', 'rig_mapper', 'sigma_road.ply')
    else:
        ply_path = join(scene, 'filtered_global_map.pcd')
    destination_folder = join(scene, "sfm", 'chouzhen', 'rig_mapper', 'local_plys', index)
    if not exists(destination_folder):
        os.mkdir(destination_folder)

    # 读取点云文件
    point_cloud = o3d.io.read_point_cloud(ply_path)
    local_ply_path = join(destination_folder, f"{index}_{type}.ply")
    # 计算点云中每个点与目标点的距离
    distances = np.asarray(point_cloud.compute_point_cloud_distance(
        o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array([position])))))

    # 找出距离小于等于指定距离的点的索引
    inliers = np.where(distances <= radius)[0]

    # 把这些点坐标都减去目标点坐标，得到相对坐标
    local_points = np.asarray(point_cloud.points)[inliers]
    local_points -= position

    local_ply = o3d.geometry.PointCloud()
    local_ply.points = o3d.utility.Vector3dVector(local_points - position)
    o3d.io.write_point_cloud(local_ply_path, local_ply)
    return local_ply_path


def get_eval(scene, radius, threshold):
    positions = get_car_positions(scene)
    output_txt = join(scene, "sfm", 'chouzhen', 'rig_mapper', 'eval.txt')
    global_average_distance_error = 0
    global_filter_point_count = 0
    count = 0
    local_plys_path = join(scene, "sfm", 'chouzhen', 'rig_mapper', 'local_plys')
    if exists(local_plys_path):
        os.system('rm -rf ' + local_plys_path)
    os.mkdir(local_plys_path)
    with open(output_txt, 'w') as f:
        for timestamp, position in positions.items():
            print(timestamp, position)
            local_sfm_ply_path = get_local_ply(scene, position, radius, timestamp, 'sfm')
            local_ref_ply_path = get_local_ply(scene, position, radius, timestamp, 'ref')
            local_average_distance_error, local_filter_point_count = compute_distance(local_ref_ply_path,
                                                                                      local_sfm_ply_path,
                                                                                      threshold)
            global_average_distance_error += local_average_distance_error
            global_filter_point_count += local_filter_point_count
            f.write(
                f"{timestamp}  local_average_distance_error:{local_average_distance_error}  local_filter_point_count:" +
                f"{local_filter_point_count}  at threshold : {threshold}\n")
            count += 1
        f.write(f"\ncount:{count}  global_average_distance_error:{global_average_distance_error/count}  global_filter_point_count:" +
                f"{global_filter_point_count/count}  at threshold : {threshold}\n")



get_eval('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-50-01_46.bag', 10, 0.5)
