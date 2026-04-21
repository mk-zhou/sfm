import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join
import shutil

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


def get_images_and_segs(scene):
    # 创建目标文件夹
    seg_folder = join(scene, 'seg')
    sky_output_folder = join(scene, 'sfm', 'chouzhen', 'seg_sky')
    dynamic_output_folder = join(scene, 'sfm', 'chouzhen', 'seg_static')
    image_list_txt = join(scene, 'sfm', 'chouzhen', 'images_list.txt')
    new_image_path=join(scene, 'sfm', 'chouzhen', 'images')
    if os.path.exists(join(scene, 'image')):
        or_image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        or_image_path = join(scene, 'rawCamera')
    else:
        or_image_path = join(scene, 'rawData')
    if os.path.exists(sky_output_folder):
        os.system('rm -rf ' + sky_output_folder)
    if os.path.exists(dynamic_output_folder):
        os.system('rm -rf ' + dynamic_output_folder)
    if os.path.exists(dynamic_output_folder):
        os.system('rm -rf ' + new_image_path)
    os.system('mkdir ' + new_image_path)
    os.system('mkdir ' + sky_output_folder)
    os.system('mkdir ' + dynamic_output_folder)
    with open(image_list_txt, 'r') as f:
        image_list = f.readlines()
    image_list = [x.strip().replace('.jpg', '_bin.png') for x in image_list if "camera-73-encoder" in x]
    print(image_list)
    count = 0  # 计数器，记录已处理的图像数量

    for i, filename in enumerate(tqdm(image_list)):
        shutil.copy(join(or_image_path,filename.replace('_bin.png','.jpg')), join(new_image_path,filename.replace('_bin.png','.jpg').split('/')[-1]))
        if filename.endswith('.png'):
            # 构建完整的文件路径
            file_path = os.path.join(seg_folder, filename)
            # 读取灰度图像
            img = Image.open(file_path).convert('L')
            # 将图像转换为 NumPy 数组
            seg_cv_array_for_dynamic = np.array(img)
            seg_cv_array_for_sky = np.array(img)
            # 获取满足条件的像素索引
            dynamic_mask = np.logical_or.reduce(
                (seg_cv_array_for_dynamic == 0, seg_cv_array_for_dynamic == 1, seg_cv_array_for_dynamic == 27,
                 np.logical_and(seg_cv_array_for_dynamic >= 19, seg_cv_array_for_dynamic <= 22),
                 np.logical_and(seg_cv_array_for_dynamic >= 52, seg_cv_array_for_dynamic <= 65))
            )
            sky_mask = (seg_cv_array_for_dynamic == 27)

            # 将满足条件的像素变成灰度值为0
            seg_cv_array_for_dynamic[dynamic_mask] = 0
            seg_cv_array_for_dynamic[~dynamic_mask] = 255
            seg_cv_array_for_sky[sky_mask] = 0
            seg_cv_array_for_sky[~sky_mask] = 255
            # print(seg_cv_array)
            # 将 NumPy 数组转换为图像对象
            dynamic_img = Image.fromarray(seg_cv_array_for_dynamic)
            sky_img = Image.fromarray(seg_cv_array_for_sky)
            # 构建新的文件路径
            mask_name = filename.replace('_bin.png', '.png')
            mask_name = mask_name.split('/')[-1]
            dynamic_file_path = os.path.join(dynamic_output_folder, mask_name)
            sky_file_path = os.path.join(sky_output_folder, mask_name)
            # 保存新生成的图像
            dynamic_img = process_grayscale_image(dynamic_img, 0.43)
            dynamic_img.save(dynamic_file_path)
            sky_img = process_grayscale_image(sky_img, 0.43)
            sky_img.save(sky_file_path)
            count += 1  # 计数器自增

def get_colmap_model(scene):
    sfm_path = join(scene, 'sfm','chouzhen')
    rig_mapper_path = join(sfm_path,'rig_mapper')
    sparse_path = join(sfm_path,'sparse','0')
    if os.path.exists(sparse_path):
        os.system('rm -rf ' + os.path.dirname(sparse_path))
    os.system('mkdir ' + os.path.dirname(sparse_path))
    os.system('mkdir ' + sparse_path)
    if os.path.exists(join(rig_mapper_path,'txt')):
        txt_path = join(rig_mapper_path,'txt')
    elif os.path.exists(join(rig_mapper_path,'0','cameras.txt')):
        txt_path = join(rig_mapper_path,'0')
    else:
        return False
    or_cameras_txt = join(txt_path,'cameras.txt')
    or_images_txt = join(txt_path,'images.txt')
    or_points3D_txt = join(txt_path,'points3D.txt')
    shutil.copy(or_points3D_txt,join(sparse_path,'points3D.txt'))
    with open(or_cameras_txt,'r') as f:
        cameras_lines = f.readlines()[3:]
    for cameras_line in cameras_lines:
        cameras_line = cameras_line.strip().split()
        if cameras_line[0] == '1':
            with open(join(sparse_path,'cameras.txt'),'a') as f:
                f.write(' '.join(cameras_line)+'\n')
    with open(or_images_txt,'r') as f:
        images_lines = f.readlines()[4:]
    with open(join(sparse_path,'images.txt'),'w') as f:
        for i,images_line in enumerate(images_lines[::2]):
            images_line = images_line.strip().split()
            if images_line[-2] == '1':
                print(images_line)
                #print(images_lines[2*i+1])
                f.write(' '.join(images_line)+'\n')
                f.write(images_lines[2*i+1])



if __name__ == '__main__':
    # seg_folder = '/data/sfm/dataset/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83/chouzhen1/seg'
    # output_folder = '/data/sfm/dataset/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83/chouzhen1/seg_road'
    #get_images_and_segs('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83')
    get_colmap_model('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83')
