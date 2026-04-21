import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
from os.path import join

def rm_tra_points(input_road, input_car, output_file, radius=20):
    pcd_road = o3d.io.read_point_cloud(input_road)
    pcd_car = o3d.io.read_point_cloud(input_car)

    # 创建K-D树以加速最近邻搜索
    kd_tree = o3d.geometry.KDTreeFlann(pcd_road)

    # 存储去噪后的路面点云
    filtered_cloud = o3d.geometry.PointCloud()

    # 存储已经加入到filtered_cloud中的点
    filtered_set = set()

    # 遍历车辆轨迹点云中的每一个点
    for point in tqdm(pcd_car.points, desc='Processing'):
        [_, indices, _] = kd_tree.search_radius_vector_3d(point, radius)

        # 检查搜索半径内的邻居点数
        if len(indices) > 0:
            for index in indices:
                filtered_point = pcd_road.points[index]
                if tuple(filtered_point) not in filtered_set:
                    filtered_cloud.points.append(filtered_point)
                    filtered_cloud.colors.append(pcd_road.colors[index])
                    # 将已经加入到filtered_cloud中的点加入到filtered_set中
                    filtered_set.add(tuple(filtered_point))
    num_points_after = len(filtered_cloud.points)
    print("过滤后的点云数量：", num_points_after)
    # 保存新的PLY文件
    o3d.io.write_point_cloud(output_file, filtered_cloud)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="")
    parser.add_argument("--sfm_folder", default="sfm")
    parser.add_argument("--dense_folder", default="dense")
    args = parser.parse_args()

    scene = args.scene
    sfm_folder = args.sfm_folder
    dense_folder = args.dense_folder

    input_road = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_rmdy_fil.ply')
    input_car = join(scene, sfm_folder, 'car.ply')
    output_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_rmdy_fil_rm40.ply')
    radius = 20

    # 移除噪点
    rm_tra_points(input_road, input_car, output_file, radius)
