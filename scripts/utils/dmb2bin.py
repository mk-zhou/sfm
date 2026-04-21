import struct
import numpy as np
from tqdm import tqdm
import os
import argparse
from os.path import join
import multiprocessing
from functools import partial
import shutil

def convert_dmb_to_bin_depth(dmb_file, bin_file):
    depth = read_dmb_depth(dmb_file)
    width, height = depth.shape  # 注意调换高度和宽度的顺序
    channels = 1  # 深度图通道数为1

    with open(bin_file, "wb") as fid:
        header = f"{height}&{width}&{channels}&"  # 调整宽度和高度的顺序
        fid.write(header.encode())  # 写入文本头部
        depth_data = np.transpose(depth, (1, 0)).astype(np.float32).tobytes(order="F")  # 转换为二进制数据并调整形状
        fid.write(depth_data)  # 写入二进制数据

def convert_dmb_to_bin_normal(dmb_file, bin_file):
    normal = read_dmb_normal(dmb_file)
    width, height, channels = normal.shape  # 注意调换高度和宽度的顺序
    channels = 3

    with open(bin_file, "wb") as fid:
        header = f"{height}&{width}&{channels}&"  # 调整宽度和高度的顺序
        fid.write(header.encode())  # 写入文本头部
        normal_data = np.transpose(normal, (1, 0, 2)).astype(np.float32).tobytes(order="F") # 转换为二进制数据并调整形状
        fid.write(normal_data)  # 写入二进制数据

def read_dmb_depth(fname):
    with open(fname, "rb") as file_handler:
        type1, = struct.unpack("i",file_handler.read(4))
        h, = struct.unpack("i",file_handler.read(4))
        w, = struct.unpack("i",file_handler.read(4))
        nb, = struct.unpack("i",file_handler.read(4))
        depth = np.fromfile(file_handler, dtype=np.float32).reshape(h, w)
    return depth

def read_dmb_normal(fname):
    with open(fname, "rb") as file_handler:
        type1, = struct.unpack("i", file_handler.read(4))
        h, = struct.unpack("i", file_handler.read(4))
        w, = struct.unpack("i", file_handler.read(4))
        nb, = struct.unpack("i", file_handler.read(4))
        normals_data = np.fromfile(file_handler, dtype=np.float32).reshape(h, w, 3)
    return normals_data

def process_dmb2bin(scene, sfm_folder='sfm'):
    acmp_dir = join(scene, sfm_folder, 'chouzhen', 'ACMP')
    bin_depth_dir = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'stereo', 'depth_maps')
    bin_normal_dir = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'stereo', 'normal_maps')
    image_list_path = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')

    os.makedirs(bin_depth_dir, exist_ok=True)  # 创建二进制文件保存目录
    os.makedirs(bin_normal_dir, exist_ok=True)  # 创建二进制文件保存目录

    entries = [entry for entry in os.scandir(acmp_dir) if entry.is_dir() and entry.name.startswith("2333_")]
    entries.sort(key=lambda x: int(x.name.split('_')[-1]))
    # print('entries',entries)
    # exit()

    # 读取images_list.txt文件
    with open(image_list_path, "r") as f:
        image_list = f.readlines()

    for idx, entry in tqdm(enumerate(entries), total=len(entries), desc="Converting files"):
        subdir = entry.name
        subdir_depth_path = os.path.join(bin_depth_dir, os.path.dirname(image_list[idx].strip()))
        subdir_normal_path = os.path.join(bin_normal_dir, os.path.dirname(image_list[idx].strip()))

        os.makedirs(subdir_depth_path, exist_ok=True)  # 创建中间文件夹
        os.makedirs(subdir_normal_path, exist_ok=True)  # 创建中间文件夹

        dmb_depth_file = os.path.join(acmp_dir, subdir, "depths.dmb")
        bin_depth_file = os.path.join(bin_depth_dir, f"{image_list[idx].strip()}.photometric.bin")
        # print('dmb_depth_file', dmb_depth_file)
        # print('bin_depth_file', bin_depth_file)
        # exit()
        convert_dmb_to_bin_depth(dmb_depth_file, bin_depth_file)

        dmb_normal_file = os.path.join(acmp_dir, subdir, "normals.dmb")
        bin_normal_file = os.path.join(bin_normal_dir, f"{image_list[idx].strip()}.photometric.bin")
        convert_dmb_to_bin_normal(dmb_normal_file, bin_normal_file)

    print("dmb2bin done")

def process_entry(entry_path, idx, image_list, bin_depth_dir, bin_normal_dir):
    subdir = os.path.basename(entry_path)
    subdir_depth_path = os.path.join(bin_depth_dir, os.path.dirname(image_list[idx].strip()))
    subdir_normal_path = os.path.join(bin_normal_dir, os.path.dirname(image_list[idx].strip()))

    os.makedirs(subdir_depth_path, exist_ok=True)
    os.makedirs(subdir_normal_path, exist_ok=True)

    dmb_depth_file = os.path.join(entry_path, "depths.dmb")
    bin_depth_file = os.path.join(bin_depth_dir, f"{image_list[idx].strip()}.photometric.bin")
    convert_dmb_to_bin_depth(dmb_depth_file, bin_depth_file)

    dmb_normal_file = os.path.join(entry_path, "normals.dmb")
    bin_normal_file = os.path.join(bin_normal_dir, f"{image_list[idx].strip()}.photometric.bin")
    convert_dmb_to_bin_normal(dmb_normal_file, bin_normal_file)


def process_dmb2bin_multiprocess(scene, sfm_folder='sfm'):
    acmp_dir = join(scene, sfm_folder, 'chouzhen', 'ACMP')
    bin_depth_dir = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'stereo', 'depth_maps')
    bin_normal_dir = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP', 'stereo', 'normal_maps')
    image_list_path = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')

    os.makedirs(bin_depth_dir, exist_ok=True)
    os.makedirs(bin_normal_dir, exist_ok=True)

    entries = [entry.path for idx, entry in enumerate(os.scandir(acmp_dir)) if
               entry.is_dir() and entry.name.startswith("2333_")]
    entries.sort(key=lambda x: int(x.split('_')[-1]))
    entries = [(path, idx) for idx, path in enumerate(entries)]

    with open(image_list_path, "r") as f:
        image_list = f.readlines()

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    list(
        tqdm(
            pool.starmap(partial(process_entry, image_list=image_list, bin_depth_dir=bin_depth_dir,
                                 bin_normal_dir=bin_normal_dir), entries),
            total=len(entries),
            desc="Converting files"
        )
    )

    pool.close()
    pool.join()

    print("dmb2bin done")

def link_file(source_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中所有子文件夹的路径
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

    # 遍历每个子文件夹，并将其中的子文件夹复制到目标文件夹中
    for subfolder in subfolders:
        # 获取子文件夹名称，即 ACMP0 到 ACMPn
        subfolder_name = os.path.basename(subfolder)
        subfolder_files = [f.path for f in os.scandir(subfolder) if f.is_dir()]

        # 遍历每个子文件夹，并将其复制到目标文件夹中
        for subfolder_file in subfolder_files:
            subfolder_file_name = os.path.basename(subfolder_file)
            destination_path = os.path.join(destination_folder, subfolder_file_name)

            # print('subfolder_file', subfolder_file)
            # print('destination_path', destination_path)
            # exit()

            # 创建软链接
            os.symlink(subfolder_file, destination_path)


def link_ACMP(scene, sfm_folder='sfm'):
    source_folder = join(scene, sfm_folder, 'chouzhen', 'ACMP_all')
    destination_folder = join(scene, sfm_folder, 'chouzhen', 'ACMP')
    link_file(source_folder, destination_folder)

def process_ACMP(scene, sfm_folder='sfm'):
    link_ACMP(scene, sfm_folder)
    process_dmb2bin_multiprocess(scene, sfm_folder)

if __name__ == "__main__":
    # process_ACMP('/vepfs_dataset/sjtu/IRMV/sfm/dataset_merge/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025607_44', 'sfm')
    # process_dmb2bin_multiprocess('/vepfs_dataset/sjtu/IRMV/sfm/dataset_merge/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025607_44', 'sfm')
    process_dmb2bin('/vepfs_dataset/sjtu/IRMV/sfm/dataset_merge/557039854_336/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025607_44', 'sfm')