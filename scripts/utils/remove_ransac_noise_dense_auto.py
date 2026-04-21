import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
from os.path import join


def rm_ransac_points(input_file, output_file, grid_width=15, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    pcd = o3d.io.read_point_cloud(input_file)

    # 获取点云xyz坐标值
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 计算点云在xy方向上的范围
    xmin, ymin, _ = np.min(points, axis=0)
    xmax, ymax, _ = np.max(points, axis=0)
    # print("点云在x方向上的范围：", xmin, xmax)
    # print("点云在y方向上的范围：", ymin, ymax)

    # 计算栅格数量
    nx = int((xmax - xmin) / grid_width) + 1
    ny = int((ymax - ymin) / grid_width) + 1
    print("栅格数量：", nx, ny)

    # 去噪后的点云
    cleaned_pcd = o3d.geometry.PointCloud()

    for i in tqdm(range(nx), desc="Processing grids"):
        for j in range(ny):
            # 确定栅格边界
            xmin_grid = xmin + i * grid_width
            ymin_grid = ymin + j * grid_width
            xmax_grid = xmin_grid + grid_width
            ymax_grid = ymin_grid + grid_width

            # 提取栅格内的点
            mask_x = np.logical_and(points[:, 0] >= xmin_grid, points[:, 0] < xmax_grid)
            mask_y = np.logical_and(points[:, 1] >= ymin_grid, points[:, 1] < ymax_grid)
            mask = np.logical_and(mask_x, mask_y)
            grid_points = o3d.geometry.PointCloud()
            grid_points.points = o3d.utility.Vector3dVector(points[mask])
            grid_colors = o3d.geometry.PointCloud()
            grid_colors.colors = o3d.utility.Vector3dVector(colors[mask])

            num_points = len(grid_points.points)
            if num_points < 30:
                # 栅格内点数小于10，则跳过该栅格
                continue

            # RANSAC平面拟合
            plane_model, inliers = grid_points.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n,
                                                             num_iterations=num_iterations)

            if len(inliers) < 10:
                # 若不能拟合出平面，则跳过该栅格内所有点
                continue

            # 保留栅格内的内点，去除外点
            cleaned_points = np.asarray(grid_points.points)[inliers]
            cleaned_colors = np.asarray(grid_colors.colors)[inliers]

            # 将清理后的点云和颜色信息添加到去噪后的点云中
            cleaned_pcd.points.extend(cleaned_points.tolist())
            cleaned_pcd.colors.extend(cleaned_colors.tolist())

    # 保存新的PLY文件
    o3d.io.write_point_cloud(output_file, cleaned_pcd)


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="")
    parser.add_argument("--sfm_folder", default="sfm")
    parser.add_argument("--dense_folder", default="dense")
    args = parser.parse_args()

    scene = args.scene
    sfm_folder = args.sfm_folder
    dense_folder = args.dense_folder'''

    '''input_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_fil.ply')
    output_file = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_ransac.ply')'''
    input_file = '/dataset/rtfbag/road_data/557039852_13/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-030228_17/sfm/chouzhen/dense_ACMP/acmap_road_fil.ply'
    output_file = '/dataset/rtfbag/road_data/557039852_13/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-030228_17/sfm/chouzhen/dense_ACMP/acmap_road_ransac.ply'

    # 调用函数进行点云去噪
    rm_ransac_points(input_file, output_file)
