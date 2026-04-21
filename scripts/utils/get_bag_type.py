import os
from os.path import join
import subprocess
import datetime

def get_seg_bag(folder_path):
    # 遍历文件夹下的所有子文件夹
    no_seg_folder = []
    only_slam_folder = []
    dr_pgo_pose_folder = []
    dr_pose_folder = []
    output_slam_txt = join(folder_path, 'slam_pose_bags.txt')
    output_dr_pgo_pose_txt = join(folder_path, 'dr_pgo_pose_bags.txt')
    output_dr_pose_txt = join(folder_path, 'dr_pose_bags.txt')
    output_no_seg_txt = join(folder_path, 'no_seg_bags.txt')
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = join(folder_path, sub_folder)
        # 判断子文件夹是否为文件夹
        if os.path.isdir(sub_folder_path):
            seg_path = join(sub_folder_path, 'seg2')
            files_path = join(sub_folder_path, 'files')
            if not os.path.isdir(join(seg_path, 'camera-83-undistort')) :
                no_seg_folder.append(sub_folder_path)
                continue
            #if not os.path.exists(files_path) and os.path.exists(join(sub_folder_path, 'global_imu_pose_INTER_aligned_time_TUM.txt')):
            if os.path.exists(join(files_path,'dr_pgo_pose_ALIGN_global_imu_pose_INTER_aligned_time_TUM.txt')):
                dr_pgo_pose_folder.append(sub_folder_path)
                continue
            if os.path.exists(join(files_path,'dr_pose_ALIGN_global_imu_pose_INTER_aligned_time_TUM.txt')):
                dr_pose_folder.append(sub_folder_path)
                continue
            if  os.path.exists(join(sub_folder_path, 'global_imu_pose_INTER_aligned_time_TUM.txt')):
                only_slam_folder.append(sub_folder_path)
                continue

            print('aaa folder: ', sub_folder_path)
    current_time = datetime.datetime.now()
    # 将时间格式化为字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_slam_txt, 'w') as f:
        f.write(formatted_time + '\n')
        f.write('slam_pose bags: ' + str(len(only_slam_folder)) + '\n')
        for folder in only_slam_folder:
            f.write(folder + '\n')
    with open(output_dr_pgo_pose_txt, 'w') as f:
        f.write(formatted_time + '\n')
        f.write('dr_pgo_pose bags: ' + str(len(dr_pgo_pose_folder)) + '\n')
        for folder in dr_pgo_pose_folder:
            f.write(folder + '\n')
    with open(output_dr_pose_txt, 'w') as f:
        f.write(formatted_time + '\n')
        f.write('dr_pose bags: ' + str(len(dr_pose_folder)) + '\n')
        for folder in dr_pose_folder:
            f.write(folder + '\n')
    with open(output_no_seg_txt, 'w') as f:
        f.write(formatted_time + '\n')
        f.write('no seg bags: ' + str(len(no_seg_folder)) + '\n')
        for folder in no_seg_folder:
            f.write(folder + '\n')

    print('no seg folder: ', len(no_seg_folder), no_seg_folder)
    print('only slam folder: ', len(only_slam_folder), only_slam_folder)
    print('dr_pgo_pose folder: ', len(dr_pgo_pose_folder), dr_pgo_pose_folder)
    print('dr_pose folder: ', len(dr_pose_folder), dr_pose_folder)



def count_dense_folders(folder_path):
    count = 0
    total_space = 0
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = join(folder_path, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue
        for sub_sub_folder in os.listdir(sub_folder_path):
            if 'sfm' not in sub_sub_folder or not os.path.isdir(join(sub_folder_path, sub_sub_folder)):
                continue
            chouzhen_path = join(sub_folder_path, sub_sub_folder, 'chouzhen')
            if not os.path.isdir(chouzhen_path):
                continue
            for sub_sub_sub_folder in os.listdir(chouzhen_path):
                if 'dense' in sub_sub_sub_folder:
                    count += 1
                    result = subprocess.run(['du', '-sh', join(chouzhen_path, sub_sub_sub_folder)], capture_output=True,
                                            text=True)
                    output = result.stdout.split()
                    print(result.stdout)
                    total_space += float(output[0][:-1])
    print(count)
    print(total_space)
    return count


get_seg_bag('/nas_gt/awesome_gt/Lane_GT_GE/appen_raw_data/20230908')
