import math
import os
import numpy as np
import open3d as o3d


def rt_trans(source_file, output_file, R, T):
    # 加载点云文件
    source_pc = o3d.io.read_point_cloud(source_file)

    # 创建变换矩阵
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # 应用变换矩阵
    transformed_pc = source_pc.transform(transformation_matrix)

    # 保存到PLY文件
    o3d.io.write_point_cloud(output_file, transformed_pc)


def get_rt_sigma_road_ply(scene):
    txt_path = os.path.join(scene, 'sfm', 'chouzhen', 'rig_mapper')
    input_ply = os.path.join(txt_path, 'sigma_road.ply')
    output_ply = os.path.join(txt_path, 'rt_sigma_road.ply')
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T = [-1.810, 0, -0.043]
    rt_trans(input_ply, output_ply, R, T)


if __name__ == '__main__':
    get_rt_sigma_road_ply('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83')
