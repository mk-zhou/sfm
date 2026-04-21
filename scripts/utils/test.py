#帮我写个脚本，把文件夹中图片大小进行resize
import os
from PIL import Image
from tqdm import tqdm
def resize_folder(folder_path):
    files = os.listdir(folder_path)
    print(files)
    for file in tqdm(files):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            # 调整大小为指定宽度和高度
            resized_image = image.resize((1920, 1080))
            # 保存调整大小后的图像
            resized_image.save(file_path)

resize_folder('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/test/images')