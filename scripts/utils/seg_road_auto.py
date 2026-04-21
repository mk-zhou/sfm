import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join
import argparse
import multiprocessing
import errno

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


def get_seg_road_folder(seg_folder, output_folder, image_list_txt):

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
            mask = ~np.isin(seg_cv_array, [3, 4, 12, 15, 16, 17, 18])
            # 将满足条件的像素变成灰度值为0
            seg_cv_array[mask] = 0
            # print(seg_cv_array)
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

def get_seg_road_folder(scene, sfm_folder='sfm', dense_folder='dense'):
    if os.path.exists(join(scene, 'seg')):
        seg_folder = join(scene, 'seg')
    else:
        seg_folder = join(scene, 'seg2')
    output_folder = join(scene, sfm_folder, 'chouzhen', 'seg_road')
    image_list_txt = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')
    get_seg_road_folder(seg_folder, output_folder, image_list_txt)
    print('get_seg_road_folder done')


def process_image_folder(seg_folder, output_folder, filename):
    try:
        if filename.endswith('.png'):
            file_path = os.path.join(seg_folder, filename)
            img = Image.open(file_path).convert('L')
            seg_cv_array = np.array(img)

            mask = ~np.isin(seg_cv_array, [3, 4, 12, 15, 16, 17, 18])

            seg_cv_array[mask] = 0
            seg_cv_array[~mask] = 255
            img = Image.fromarray(seg_cv_array)

            seg_filename = filename.replace('_bin.png', '.jpg.png')
            new_file_path = os.path.join(output_folder, seg_filename)
            new_folder_path = os.path.dirname(new_file_path)

            if not os.path.exists(new_folder_path):
                try:
                    os.makedirs(new_folder_path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

            if img.size != (960, 540):
                img = img.resize((960, 540))

            if filename.startswith('camera-0'):
                img = process_grayscale_image(img, 0.28)

            img.save(new_file_path)
            # print('Saved:', new_file_path)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

def get_seg_road_folder_auto(scene, sfm_folder='sfm', dense_folder='dense'):
    if os.path.exists(join(scene, 'seg')):
        seg_folder = join(scene, 'seg')
    else:
        seg_folder = join(scene, 'seg2')
    # seg_folder = join(scene, 'seg2')
    output_folder = join(scene, sfm_folder, 'chouzhen', 'seg_road')
    image_list_txt = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(image_list_txt, 'r') as f:
        image_list = f.readlines()
    image_list = [x.strip().replace('.jpg', '_bin.png') for x in image_list]
    # print('image_list', image_list)
    # print('image_list', len(image_list))
    # exit()
    pool = multiprocessing.Pool()

    for i, filename in enumerate(image_list):
        pool.apply_async(process_image_folder, (seg_folder, output_folder, filename))

    pool.close()
    pool.join()
    print('get_seg_road_folder done')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--scene", default="")
    # parser.add_argument("--sfm_folder", default="sfm")
    # parser.add_argument("--dense_folder", default="dense")
    # args = parser.parse_args()
    #
    # scene = args.scene
    # sfm_folder = args.sfm_folder
    # dense_folder = args.dense_folder
    #
    # # 创建目标文件夹
    # seg_folder = join(scene, 'seg2')
    # output_folder = join(scene, sfm_folder, 'chouzhen', 'seg_road_half')
    # image_list_txt = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')

    get_seg_road_folder_auto('/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/data/EP41-ORIN-0S14-00G_20240523_081429/EP41-ORIN-0S14-00G_FULL_MCAP_02.22.03_20240414-102825_25.mcap', 'sfm_seg2_new')
    # get_seg_road_folder_auto_multiprocess('/dataset/rtfbag/merge_datasets/557039854_190/has_go/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-022113_16')
