import os
import cv2
from os.path import join
from tqdm import tqdm

def get_video(scene):
    if os.path.exists(join(scene, 'image')):
        image_folder_path = join(scene, 'image', 'camera-0-undistort')
        #image_folder_path = join(scene, 'image', 'camera-73-encoder-undistort')
        #image_folder_path = join(scene, 'image', 'soc_encoded_camera_0-undistort')
        #image_folder_path = join(scene, 'image', 'soc_encoded_camera_1-undistort')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_folder_path = join(scene, 'rawCamera', 'camera-73-undistort')
    else:
        image_folder_path = join(scene, 'rawData', 'camera-73-undistort')
    image_files = sorted(os.listdir(image_folder_path))
    first_image_path = os.path.join(image_folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    output_video_path = join(scene, 'video.avi')
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

    # 将图片逐帧写入视频
    for image_file in tqdm(image_files[::5]):
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    # 释放资源
    video_writer.release()


if __name__ == '__main__':
    scene = '/dataset/handsome_man_with_handsome_data/sfm_test/input/1'
    get_video(scene)
