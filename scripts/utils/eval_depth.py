import os
from PIL import Image
import numpy as np
from os.path import join


def calculate_depth_error(project_path):
    # 获取文件夹中的所有图像文件
    images1 = [f for f in os.listdir(join(project_path, 'depth')) if f.endswith('.npy')]
    images2 = [f for f in os.listdir(join(project_path, 'sfm', 'chouzhen', 'depth')) if f.endswith('.npy')]

    # 确保两个文件夹中的图像数量相同
    if len(images1) != len(images2):
        print("两个文件夹中的图像数量不匹配！")
        return

    # 遍历每对图像并计算深度误差
    total_error = 0.0
    num_pixels = 0
    output_path = join(project_path, 'depth_errors')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for img1_name, img2_name in zip(images1, images2):
        img1_path = join(project_path, 'depth', img1_name)
        img2_path = join(project_path, 'sfm', 'chouzhen', 'depth', img2_name)

        # 打开图像并将其转换为灰度图
        arr1 = np.load(img1_path)
        arr2 = np.load(img2_path)

        # 把arr1-arr2保存成深度图，如果一个点在两个图像中都有值，则保存该点的深度差，否则保存255
        '''mask1 = np.logical_or(arr1 == 255, arr2 == 255)
        depth_diff = np.abs(arr1 - arr2)
        depth_diff[mask1] = 0
        depth_diff = Image.fromarray(depth_diff)
        depth_diff.save(join(project_path, 'depth_diff', img1_name))'''
        # 计算有像素点的深度误差
        mask = np.logical_and(arr1 != 255, arr2 != 255)
        print(arr1[mask])
        print(arr2[mask])
        print(np.abs(arr1[mask] - arr2[mask]))

        depth_diff = np.abs(arr1[mask] - arr2[mask]) / (arr1[mask] + 1e-6)
        # 把depth_diff转换成图像保存
        depth_diff_png = np.abs(arr1 - arr2) / (arr1 + 1e-6)
        # 不在mask中的点设置为255
        mask_ = np.logical_not(mask)
        depth_diff_png[mask_] = 255
        depth_diff_png = np.array(depth_diff_png, dtype=np.uint8)
        depth_diff_img = Image.fromarray(depth_diff_png)
        depth_diff_img.save(join(output_path, img1_name.replace('.npy', '.png')))
        print(depth_diff)
        error = np.mean(depth_diff)
        print("图像{}的深度误差：{}".format(img1_name, error))
        # 更新总误差和像素计数
        total_error += sum(depth_diff)
        num_pixels += depth_diff.shape[0]

    # 计算平均深度误差
    avg_error = total_error / num_pixels

    return avg_error


depth_error = calculate_depth_error(
    '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag')
print("平均深度误差：", depth_error)
