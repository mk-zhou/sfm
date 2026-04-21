import numpy as np
import cv2

# 从二进制文件中加载深度图
depth_map = np.fromfile(
    '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm/chouzhen/dense/stereo/depth_maps/camera-73-undistort/1690465684.433333158.jpg.photometric.bin',
    dtype=np.float32).reshape(1920, 1080)

# 将深度图进行归一化，以便在可视化时显示
normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 保存可视化的深度图
cv2.imwrite('test.jpg', normalized_depth_map)
