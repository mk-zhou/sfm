import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join
import argparse


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


def get_seg_dynamic_folder(seg_folder, output_folder, image_list_txt):

    if not os.path.exists(output_folder):
        os.system('mkdir ' + output_folder)
    with open(image_list_txt, 'r') as f:
        image_list = f.readlines()
    image_list = [x.strip().replace('.jpg', '_bin.png') for x in image_list]
    # print(image_list)

    for i, filename in enumerate(tqdm(image_list)):
        if filename.endswith('.png'):
            # 构建完整的文件路径
            file_path = os.path.join(seg_folder, filename)

            # 读取灰度图像
            img = Image.open(file_path).convert('L')

            # 将图像转换为 NumPy 数组
            seg_cv_array = np.array(img)

            # 获取满足条件的像素索引
            mask = np.isin(seg_cv_array, [0, 1, 9, 13, 14])
            # 将满足条件的像素变成灰度值为0
            seg_cv_array[mask] = 0
            # 将 NumPy 数组转换为图像对象
            img = Image.fromarray(seg_cv_array)

            # 构建新的文件路径
            # print("filename", filename)
            seg_filename = filename.replace('_bin.png', '.jpg.png')
            # print("seg_filename", seg_filename)
            new_file_path = os.path.join(output_folder, seg_filename)
            # print("new_file_path", new_file_path)


            new_folder_path = os.path.dirname(join(output_folder, filename))
            # print("new_folder_path", new_folder_path)
            # exit()
            if not os.path.exists(new_folder_path):
                os.system('mkdir ' + new_folder_path)
            # 保存新生成的图像
            if img.size != (960, 540):
                if img.size == (3840, 2160):
                    img = img.resize((960, 540))
                    img = process_grayscale_image(img, 0.43)
                else:
                    img = img.resize((960, 540))
            img.save(new_file_path)

def seg_dy(scene, sfm_folder='sfm'):
    if os.path.exists(join(scene, 'seg')):
        seg_folder = 'seg'
    else:
        seg_folder = 'seg2'
    seg_path = join(scene, seg_folder)
    output_path = join(scene, sfm_folder, 'chouzhen', 'seg_dynamic')
    image_list_txt = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')
    get_seg_dynamic_folder(seg_path, output_path, image_list_txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="")
    args = parser.parse_args()

    # 创建目标文件夹
    seg_folder = join(args.scene, 'seg')
    output_folder = join(args.scene, 'sfm', 'chouzhen', 'seg_dynamic')
    image_list_txt = join(args.scene, 'sfm', 'chouzhen', 'images_list.txt')

    # seg_folder = '/dataset/rtfbag/merge_datasets/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025607_44/seg2'
    # output_folder = '/dataset/rtfbag/merge_datasets/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025607_44/sfm/chouzhen/seg_dynamic'
    # image_list_txt = '/dataset/rtfbag/merge_datasets/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025607_44/sfm/chouzhen/images_list.txt'

    get_seg_dynamic_folder(seg_folder, output_folder, image_list_txt)
    # get_seg_dynamic_folder('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-14-22-03_110.bag')
