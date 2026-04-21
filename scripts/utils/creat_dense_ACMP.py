import os
from PIL import Image
from tqdm import tqdm
from os.path import join
import argparse
import multiprocessing

def process_cameras_file(input_file, output_file):
    # 读取原始文件内容
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 判断输出文件所在目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每行数据，并写入新文件
    with open(output_file, 'w') as file:
        for line in lines:
            if line.startswith("#") or line.startswith("Number of cameras"):
                # 注释行和摄像头数量行直接写入新文件
                file.write(line)
            else:
                # 拆分一行数据
                data = line.strip().split()
                camera_id, model, width, height, *params = data

                # 对应数据除以2
                width = str(int(float(width)/2))
                height = str(int(float(height)/2))
                params = [str(float(p)/2) for p in params]

                # 合并为一行数据，并写入新文件
                new_line = " ".join([camera_id, model, width, height] + params) + "\n"
                file.write(new_line)


def resize_images(input_folder, output_folder, target_width, target_height):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的每个子文件夹
    for foldername in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, foldername)
        if os.path.isdir(folder_path):
            output_subfolder = os.path.join(output_folder, foldername)
            os.makedirs(output_subfolder, exist_ok=True)

            # 遍历子文件夹中的图片文件，并添加进度条
            for filename in tqdm(os.listdir(folder_path)):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, filename)
                    output_path = os.path.join(output_subfolder, filename)

                    # 打开并resize图片
                    with Image.open(image_path) as img:
                        resized_img = img.resize((target_width, target_height))
                        resized_img.save(output_path)
def cp_file(scene, sfm_folder='sfm', dense_folder='dense'):
    sparse_imagestxt = join(scene, sfm_folder, 'chouzhen', 'rig_mapper', '0', 'images.txt')
    sparse_points3dtxt = join(scene, sfm_folder, 'chouzhen', 'rig_mapper', '0', 'points3D.txt')
    acmp_sparse_path = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'sparse')

    stereo_fu_cfg = join(scene, sfm_folder, 'chouzhen', dense_folder, 'stereo', 'fusion.cfg')
    stereo_patch_cfg = join(scene, sfm_folder, 'chouzhen', dense_folder, 'stereo', 'patch-match.cfg')
    acmp_stereo_path = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'stereo')

    cp_str1 = "cp " + sparse_imagestxt + ' ' + sparse_points3dtxt + ' ' + acmp_sparse_path
    cp_str2 = "cp " + stereo_fu_cfg + ' ' + stereo_patch_cfg + ' ' + acmp_stereo_path

    os.system(cp_str1)
    os.system(cp_str2)

def delete_file(scene, sfm_folder='sfm', dense_folder='dense'):
    acmp_path = join(scene, sfm_folder, 'chouzhen', 'ACMP')
    dense_path = join(scene, sfm_folder, 'chouzhen', dense_folder)
    cams_path = join(scene, sfm_folder, 'chouzhen', 'cams')
    acmp_imgs_path = join(scene, sfm_folder, 'chouzhen', 'images')

    rm_str = "rm -rf " + acmp_path + ' ' + dense_path + ' ' + cams_path + ' ' + acmp_imgs_path
    os.system(rm_str)

def process_dense_ACMP_all(scene, sfm_folder='sfm', dense_folder='dense'):
    # 调用函数处理文件
    input_file = join(scene, sfm_folder, 'chouzhen', 'rig_mapper', '0', 'cameras.txt')
    output_path = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'sparse')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cp_str1 = "cp " + input_file + ' ' + output_path
    os.system(cp_str1)
    # process_cameras_file(input_file, output_file)

    # 调用函数处理图片文件夹
    input_images = join(scene, sfm_folder, 'chouzhen', dense_folder, 'images')
    output_images = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP')
    cp_str2 = "cp -r " + input_images + ' ' + output_images
    os.system(cp_str2)

    # resize_images(input_images, output_images, 960, 540)
    # 调用函数复制文件
    cp_file(scene, sfm_folder, dense_folder)

    # 调用函数删除不需要的文件文件
    # delete_file(scene, sfm_folder, dense_folder)


    print('process_dense_ACMP done')

def process_dense_ACMP(scene, sfm_folder='sfm', dense_folder='dense'):
    # 调用函数处理文件
    input_file = join(scene, sfm_folder, 'chouzhen', 'rig_mapper', '0', 'cameras.txt')
    output_file = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'sparse', 'cameras.txt')
    process_cameras_file(input_file, output_file)

    # 调用函数处理图片文件夹
    input_images = join(scene, sfm_folder, 'chouzhen', dense_folder, 'images')
    output_images = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'images')
    resize_images(input_images, output_images, 960, 540)

    # 调用函数复制文件
    cp_file(scene, sfm_folder, dense_folder)

    # 调用函数删除不需要的文件文件
    # delete_file(scene, sfm_folder, dense_folder)
    print('process_dense_ACMP done')

def resize_and_save_image(image_path, output_path, width, height):
    with Image.open(image_path) as img:
        resized_img = img.resize((width, height))
        resized_img.save(output_path)

def process_dense_ACMP_multiprocess(scene, sfm_folder='sfm', dense_folder='dense'):
    # 调用函数处理文件
    input_file = join(scene, sfm_folder, 'chouzhen', 'rig_mapper', '0', 'cameras.txt')
    output_file = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'sparse', 'cameras.txt')
    process_cameras_file(input_file, output_file)

    # 调用函数处理图片文件夹
    input_images = join(scene, sfm_folder, 'chouzhen', dense_folder, 'images')
    output_images = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'images')

    # 创建进程池
    pool = multiprocessing.Pool()

    for foldername in os.listdir(input_images):
        folder_path = os.path.join(input_images, foldername)
        if os.path.isdir(folder_path):
            output_subfolder = os.path.join(output_images, foldername)
            os.makedirs(output_subfolder, exist_ok=True)

            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, filename)
                    output_path = os.path.join(output_subfolder, filename)

                    # 使用进程池异步处理每个图片文件
                    pool.apply_async(resize_and_save_image, (image_path, output_path, 960, 540))

    # 关闭进程池，等待所有进程完成
    pool.close()
    pool.join()

    # 调用函数复制文件
    cp_file(scene, sfm_folder, dense_folder)

    # 调用函数删除不需要的文件文件
    # delete_file(scene, sfm_folder, dense_folder)

    print('process_dense_ACMP done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="")
    parser.add_argument("--sfm_folder", default="sfm")
    parser.add_argument("--dense_folder", default="dense")
    args = parser.parse_args()

    scene = args.scene
    sfm_folder = args.sfm_folder
    dense_folder = args.dense_folder

    # process_dense_ACMP('/dataset/sfm/xyq/test/test_acm/42_multi')
    # process_dense_ACMP_all('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag',sfm_folder='sfm_pair_test')
    process_dense_ACMP_multiprocess('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-31_43.bag',sfm_folder='sfm_pair_test')
