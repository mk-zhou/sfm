import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join


# 处理front_wide下方的横条
def process_grayscale_image(image, ratio):
    image_array = np.array(image)
    # 获取图像的尺寸
    height, width = image_array.shape

    # 计算需要滤掉的像素行数
    filter_rows = int(height * ratio)

    # 将底部的像素点置为0
    image_array[height - filter_rows:, :] = 0
    new_image = Image.fromarray(image_array)
    return new_image


def get_seg_road_folder(scene):
    # 创建目标文件夹
    seg_folder = join(scene, 'seg')
    output_folder = join(scene, 'sfm', 'chouzhen', 'seg_road')
    image_list_txt = join(scene, 'sfm', 'chouzhen', 'images_list.txt')
    if os.path.exists(output_folder):
        os.system('rm -rf ' + output_folder)
    os.system('mkdir ' + output_folder)
    with open(image_list_txt, 'r') as f:
        image_list = f.readlines()
    image_list = [x.strip().replace('.jpg', '_bin.png') for x in image_list]
    print(image_list)
    count = 0  # 计数器，记录已处理的图像数量

    for i, filename in enumerate(tqdm(image_list)):
        if filename.endswith('.png'):
            # 构建完整的文件路径
            file_path = os.path.join(seg_folder, filename)

            # 读取灰度图像
            img = Image.open(file_path).convert('L')

            # 将图像转换为 NumPy 数组
            seg_cv_array = np.array(img)

            # 获取满足条件的像素索引
            mask = ~np.isin(seg_cv_array, [3, 4, 12, 15, 16, 17, 18])
            # seg_cv_array[:, :] = 255
            # print(seg_cv_array)
            # 将满足条件的像素变成灰度值为0
            seg_cv_array[mask] = 0
            # print(seg_cv_array)
            # 将 NumPy 数组转换为图像对象
            img = Image.fromarray(seg_cv_array)

            # 构建新的文件路径
            mask_name = filename.replace('_bin.png', '.jpg.png')
            new_file_path = os.path.join(output_folder, mask_name)
            new_folder_path = os.path.dirname(join(output_folder, filename))
            if not os.path.exists(new_folder_path):
                os.system('mkdir ' + new_folder_path)
            # 保存新生成的图像
            if img.size != (1920, 1080):
                img = img.resize((1920, 1080))
                img = process_grayscale_image(img, 0.43)
            img.save(new_file_path)

            count += 1  # 计数器自增


if __name__ == '__main__':
    # seg_folder = '/data/sfm/dataset/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83/chouzhen1/seg'
    # output_folder = '/data/sfm/dataset/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83/chouzhen1/seg_road'
    get_seg_road_folder('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83')
