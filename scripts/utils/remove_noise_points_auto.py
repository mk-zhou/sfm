import numpy as np
import open3d as o3d
from os.path import join
import argparse
from tqdm import tqdm

def remove_noise_points(input_file, output_file, radius=0.3):
    pcd = o3d.io.read_point_cloud(input_file)
    # 将点云转换为numpy数组
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # 获取点云颜色

    # 利用kdtree查找半径范围内的点
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    new_xyz = []  # 存储新的点坐标
    new_colors = []  # 存储新的点颜色

    for i in tqdm(range(xyz.shape[0])):
        [_, idx, _] = kdtree.search_radius_vector_3d(xyz[i], radius)
        if len(idx) < 5:
            # 如果半径范围内少于5个点则删除
            continue

        new_xyz.append(xyz[i])
        new_colors.append(colors[i])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(new_xyz))
    new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(new_colors))
    # 保存新的PLY文件
    o3d.io.write_point_cloud(output_file, pcd)

if __name__ == '__main__':
    '''parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="")
    parser.add_argument("--sfm_folder", default="sfm")
    parser.add_argument("--dense_folder", default="dense")
    args = parser.parse_args()

    scene = args.scene
    sfm_folder = args.sfm_folder
    dense_folder = args.dense_folder
    # 读取PLY文件
    input_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'test.ply')

    output_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'test_fil.ply')'''
    input_file = '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm/chouzhen/dense/test.ply'
    output_file = '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm/chouzhen/dense/test_fil.ply'

    # 移除噪点
    remove_noise_points(input_file, output_file)
