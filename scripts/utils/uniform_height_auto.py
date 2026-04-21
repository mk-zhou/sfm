import numpy as np
import open3d as o3d
import argparse
from os.path import join
from tqdm import tqdm

def grid_uni_Z(input_file, output_file, grid_size = 2):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(input_file)
    # 获取点云坐标
    points = np.asarray(pcd.points)
    # 计算点云的边界
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    # 计算栅格数目
    grid_nums = np.ceil((max_bound - min_bound) / grid_size).astype(int)

    # print("X轴最小值:", min_bound[0])
    # print("X轴最大值:", max_bound[0])
    # print("Y轴最小值:", min_bound[1])
    # print("Y轴最大值:", max_bound[1])
    print("拍扁栅格数目:", grid_nums[0] * grid_nums[1])
    # 创建栅格
    grids = [[] for _ in range(grid_nums[0] * grid_nums[1])]
    # 将点云分配到栅格中
    for i, point in enumerate(points):
        x_idx = int((point[0] - min_bound[0]) // grid_size)
        y_idx = int((point[1] - min_bound[1]) // grid_size)
        grid_idx = y_idx * grid_nums[0] + x_idx
        grids[grid_idx].append(point)

    # 筛选非空栅格
    filtered_grids = [grid for grid in grids if len(grid) > 0]
    # 计算每个栅格内点的z值平均
    for i, grid in enumerate(tqdm(filtered_grids)):
        avg_z = np.mean(np.asarray(grid)[:, 2])
        for j in range(len(grid)):
            grid[j][2] = avg_z

    # 更新点云
    filtered_points = np.concatenate(filtered_grids, axis=0)
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    # 保存为新的PLY文件
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

    input_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_rm8.ply')
    # input_file = join(args.scene, 'sfm', 'chouzhen', 'dense', 'dense_road_raw.ply')
    output_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_rm8_uniZ.ply')'''
    input_file = '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm/chouzhen/rig_mapper/road_ransac.ply'
    output_file = '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm/chouzhen/rig_mapper/road_ransac_uniZ.ply'
    # 设置栅格尺寸
    grid_size = 0.5

    # 进行栅格滤波
    grid_uni_Z(input_file, output_file, grid_size)

