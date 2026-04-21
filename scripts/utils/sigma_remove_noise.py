import open3d as o3d
import numpy as np
from os.path import join
from tqdm import tqdm


# 输入：未去噪的点云路径，去噪后的点云保存路径，
# x范围（前进方向距离 to do 结合car_pose，判断前进方向），
# z阈值（高度方向），y阈值（左右方向），全局阈值（用于去除过远的点）
def remove_noise_from_ply(input_file, output_file, x_range=5.0, z_threshold=0.2, y_threshold=3, global_threshold=5):
    # 读取PLY文件
    mesh = o3d.io.read_triangle_mesh(input_file)

    # 提取点云的坐标和颜色
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    # 计算全局范围内的Z坐标均值和标准差
    global_x_coords = points[:, 0]
    global_x_mean = np.mean(global_x_coords)
    global_x_std = np.std(global_x_coords)
    global_y_coords = points[:, 1]
    global_y_mean = np.mean(global_y_coords)
    global_y_std = np.std(global_y_coords)
    global_z_coords = points[:, 2]
    global_z_mean = np.mean(global_z_coords)
    global_z_std = np.std(global_z_coords)
    # 初始化过滤后的点云列表
    filtered_points = []
    filtered_colors = []

    with tqdm(total=len(points), desc="Processing", unit="point") as pbar:
        for i in range(len(points)):
            point = points[i]
            x = point[0]

            # 获取局部范围内的点
            local_points = points[(points[:, 0] >= x - x_range) & (points[:, 0] <= x + x_range)]

            if len(local_points) > 0:
                # 计算局部范围内的Z坐标均值和标准差
                z_coords = local_points[:, 2]
                z_mean = np.mean(z_coords)
                z_std = np.std(z_coords)

                y_coords = local_points[:, 1]
                y_mean = np.mean(y_coords)
                y_std = np.std(y_coords)

                # 根据局部范围内的3σ阈值去除噪声点
                if np.abs(point[2] - z_mean) <= z_threshold * z_std \
                        and np.abs(point[1] - y_mean) <= y_threshold * y_std \
                        and np.abs(point[0] - global_x_mean) <= global_threshold * global_x_std \
                        and np.abs(point[1] - global_y_mean) <= global_threshold * global_y_std \
                        and np.abs(point[2] - global_z_mean) <= global_threshold * global_z_std \
                        :
                    filtered_points.append(point)
                    filtered_colors.append(colors[i])
            pbar.update(1)

    # 创建过滤后的点云
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(filtered_points))
    filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(filtered_colors))

    # 保存去噪后的点云为PLY文件
    o3d.io.write_triangle_mesh(output_file, filtered_mesh)


# 去掉过于稀疏的点
# 输入：未去稀疏的点云路径，去稀疏后的点云保存路径，
# 半径（范围），最小邻居数量
def remove_sparse_points(input_file, output_file, radius=1.0, min_neighbor_count=5):
    # 读取PLY文件
    mesh = o3d.io.read_triangle_mesh(input_file)

    # 提取点云的坐标和颜色
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    # 初始化过滤后的点云列表
    filtered_points = []
    filtered_colors = []

    with tqdm(total=len(points), desc="Processing", unit="point") as pbar:
        for i in range(len(points)):
            point = points[i]

            # 计算点与其他点之间的距离
            distances = np.linalg.norm(points - point, axis=1)

            # 统计一米范围内的点的数量
            neighbor_count = np.sum(distances <= radius)

            # 根据最小邻居数量判断是否删除点
            if neighbor_count >= min_neighbor_count:
                filtered_points.append(point)
                filtered_colors.append(colors[i])

            pbar.update(1)

    # 创建过滤后的点云
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(filtered_points))
    filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(filtered_colors))

    # 保存去噪后的点云为PLY文件
    o3d.io.write_triangle_mesh(output_file, filtered_mesh)


def get_sigma_road_ply(scene, z_threshold=0.5, y_threshold=2.5, global_threshold=6):
    input_file = join(scene, 'sfm', 'chouzhen', 'rig_mapper',  'road.ply')  # 输入PLY文件路径
    output_file = join(scene, 'sfm', 'chouzhen', 'rig_mapper',  'sigma_road.ply')  # 输出去噪后的PLY文件路径
    remove_noise_from_ply(input_file, output_file, 5, z_threshold, y_threshold, global_threshold)
    remove_sparse_points(output_file, output_file, 1, 5)


if __name__ == '__main__':
    get_sigma_road_ply('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83')
