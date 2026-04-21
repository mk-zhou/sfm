from SuperPointDetectors import get_super_points_from_scenes_return
from SuperPointDetectors import get_sift_features_from_scenes
from SuperPointDetectors import get_super_glue_from_scenes_return
from SuperPointDetectors import SuperGlueMatching
from matchers import mutual_nn_matcher
from matchers import mut_nn_matcher_indices_0
from matchers import mut_nn_matcher_indices_1
import cv2
import os, time
import numpy as np
import argparse
from database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
from transform_colmap_camera import camTodatabase
from add_prior import add_all_prior
import torch
from tqdm import tqdm
from get_match_list import get_match_list
from merge_get_match_list import merge_get_match_list
from scipy.spatial.transform import Rotation
from turn_around_get_match_list import turn_around_get_match_list

camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10}
torch.set_grad_enabled(False)


def get_init_cameraparams(width, height, modelId):
    f = max(width, height) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId == 2 or modelId == 6:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 8:
        return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def init_cameras_database(db, images_list_path, single_camera):
    print("init cameras database ......................................")
    images_name = []
    width = None
    height = None
    with open(images_list_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces and newline characters
        if 'jpg' in line or 'png' in line:
            images_name.append(line)
    cameraModel = 'PINHOLE'
    params = get_init_cameraparams(1920, 1080, cameraModel)

    if single_camera:
        camera_id = 1
        db.add_camera(cameraModel, 1920, 1080, params, camera_id=camera_id)
        for i, name in enumerate(images_name):
            db.add_image(name, camera_id, image_id=i + 1)
    else:
        for i, name in enumerate(images_name):
            camera_id = i + 1
            db.add_camera(cameraModel, 1920, 1080, params, camera_id=camera_id)
            db.add_image(name, camera_id, image_id=i + 1)
    return images_name


##########xyq
def draw_keypoints_on_image(image, keypoints):
    # 创建一个空的三维数组用于绘制特征点
    img_with_keypoints = np.copy(image)

    # 在图像上绘制特征点
    for point in keypoints:
        x, y = point[:2]
        cv2.circle(img_with_keypoints, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绘制绿色的圆形标记在特征点位置

    # 返回带有特征点的图像
    return img_with_keypoints


########################

def import_feature(db, image_path, images_name):
    print("feature extraction by super points ...........................")
    sps = get_super_points_from_scenes_return(image_path, images_name)
    # print("sps:", sps)
    # sps = get_sift_features_from_scenes(images_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")

    new_sps = {}
    keypoints_list = []
    for i, name in enumerate(images_name):
        raw_keypoints = sps[name]['keypoints']
        raw_descriptors = sps[name]['descriptors']
        raw_scores = sps[name]['scores']
        # image_size = sps[name]['image_size']

        ###########################xyq######读取对应的灰度图像
        "后缀   _bin.png"
        # 获取文件名和文件扩展名
        '''path_components = name.split("/")
        print("path_components:", path_components)
        image_name_gray = path_components[0]+'-or/'+path_components[1].replace(".jpg", "_bin.png")'''
        image_name_gray = name.replace(".jpg", "_bin.png")
        image_path_gray = os.path.join(image_path, image_name_gray)
        img_folder_names = ['image', 'rawCamera', 'rawData']
        for img_folder_name in img_folder_names:
            if img_folder_name in image_path_gray:
                if os.path.exists(image_path_gray.replace(img_folder_name, 'seg2')):
                    image_path_gray = image_path_gray.replace(img_folder_name, 'seg2')
                else:
                    image_path_gray = image_path_gray.replace(img_folder_name, 'seg')
                break
        # image_path_gray = os.path.join(image_path, '..', 'seg', image_name_gray)
        # print("image_path_gray:", image_path_gray)
        # print("image_path_gray:", image_path_gray)
        gray_image = cv2.imread(image_path_gray, cv2.IMREAD_GRAYSCALE)
        if gray_image.shape != (1920, 1080):
            gray_image = cv2.resize(gray_image, (1920, 1080))
            # 让前视广角的gray_image的下方43%都不行
            if 'camera-0' in image_path_gray:
                gray_image[1080 - int(1080 * 0.28):, :] = 0
        # 使用布尔索引直接过滤关键点
        x = raw_keypoints[:, 0].astype(int)
        y = raw_keypoints[:, 1].astype(int)
        pixel_values = gray_image[y, x]
        # new segment label
        valid_indices = ~np.isin(pixel_values, [0, 1, 9, 13, 14])
        # old segment label
        # valid_indices = ~np.isin(pixel_values, [53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 1, 9, 13, 14, 0, 1, 19, 20, 21, 22, 34, 52, 57, 27, 64, 65])
        if np.any(valid_indices):  # 如果存在符合要求的关键点
            keypoints = raw_keypoints[valid_indices]
            descriptors = raw_descriptors[valid_indices]
            scores = raw_scores[valid_indices]
            image = sps[name]['image']
        else:  # 如果没有符合要求的关键点，则使用第一个关键点和描述子
            keypoints = np.expand_dims(raw_keypoints[0], axis=0)
            descriptors = np.expand_dims(raw_descriptors[0], axis=0)
            scores = np.array([raw_scores[0]])
            image = sps[name]['image']
        # keypoints = raw_keypoints
        # descriptors = raw_descriptors
        # scores = raw_scores

        # 将过滤后的关键点和描述子添加到new_sps中
        # new_sps[name] = {'keypoints': keypoints, 'descriptors': descriptors}
        new_sps[name] = {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores, 'image': image}
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
                                    np.ones((n_keypoints, 1)).astype(np.float32),
                                    np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i + 1, keypoints)
        # db.add_descriptors(i + 1, descriptors)
        keypoints_list.append((raw_keypoints, keypoints, name))
    return new_sps


def match_features_raw_raw(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    # sequential match
    num_images = len(images_name)
    num_per_view = num_images // 6
    step_range = list(range(1, 6)) + list(range(num_per_view + 1, num_per_view + 6)) + list(
        range(2 * num_per_view + 1, 2 * num_per_view + 6)) + list(
        range(3 * num_per_view + 1, 3 * num_per_view + 6)) + list(
        range(4 * num_per_view + 1, 4 * num_per_view + 6)) + list(range(5 * num_per_view + 1, 5 * num_per_view + 6))
    print("num_per_view:", num_per_view)
    print("step_range:", step_range)

    match_list = open(match_list_path, 'w')
    for step in step_range:
        for i in range(0, num_images - step):
            match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[i + step]]['descriptors'] * 1.0
            # Check if descriptors are not empty before calling mutual_nn_matcher
            if D1.shape[0] > 0 and D2.shape[0] > 0:
                matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
                # print("matches:", matches)
                # exit()
                db.add_matches(i + 1, i + step + 1, matches)
            else:
                print("Warning: Descriptors for images", images_name[i], "and", images_name[i + step], "are empty.")

    match_list.close()


def match_features_raw(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    # sequential match
    num_images = len(images_name)
    num_per_view = num_images // 6
    print("num_per_view:", num_per_view)
    match_list = open(match_list_path, 'w')
    step_range = []

    # print("sps_k:", sps[images_name[1]]['keypoints'].dtype)
    # print("sps_d:", sps[images_name[1]]['descriptors'].shape)
    # print("images_name:", len(images_name))
    # exit()
    for i in tqdm(range(0, num_images)):
        if i < num_per_view:
            step_range = list(range(1, 6)) + list(range(num_per_view, num_per_view + 6)) + list(
                range(2 * num_per_view, 2 * num_per_view + 6))
        elif i >= num_per_view and i < 2 * num_per_view:
            step_range = list(range(1, 6)) + list(range(2 * num_per_view, 2 * num_per_view + 6))
        elif i >= 2 * num_per_view and i < 3 * num_per_view:
            step_range = list(range(1, 6)) + list(range(2 * num_per_view, 2 * num_per_view + 6))
        elif i >= 3 * num_per_view and i < 4 * num_per_view:
            step_range = list(range(1, 6)) + list(range(2 * num_per_view, 2 * num_per_view + 6))
        elif i >= 4 * num_per_view and i < 5 * num_per_view:
            step_range = list(range(1, 6)) + list(range(num_per_view, num_per_view + 6))
        elif i >= 5 * num_per_view:
            step_range = list(range(1, 6))
        for step in step_range:
            if i + step >= num_images:
                continue
            match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[i + step]]['descriptors'] * 1.0
            print("D1", D1.shape)
            print("D2", D2.shape)
            # Check if descriptors are not empty before calling mutual_nn_matcher
            if D1.shape[0] > 0 and D2.shape[0] > 0:
                matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
                db.add_matches(i + 1, i + step + 1, matches)
            else:
                print("Warning: Descriptors for images", images_name[i], "and", images_name[i + step], "are empty.")
    match_list.close()


def match_features_first(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    # sequential match

    num_images = len(images_name)
    step_range = list(range(1, num_images))
    match_list = open(match_list_path, 'w')
    for step in step_range:
        for i in range(0, num_images - step):
            match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[i + step]]['descriptors'] * 1.0
            matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
            db.add_matches(i + 1, i + step + 1, matches)
    match_list.close()


def match_features(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    folder_image_name_list = []
    cameras = set()  # 用集合来保存不重复的视角
    for i, name in enumerate(images_name):
        folder_image_name = f"{name[0]}/{name[1]}"
        folder_image_name_list.append(folder_image_name)
        camera = name[0]
        cameras.add(camera)  # 将视角添加到集合中

    num_views = len(cameras)
    print("cameras:", cameras)
    print("Number of camera views:", num_views)

    num_images = len(images_name)
    num_per_view = num_images // num_views
    print("num_per_view:", num_per_view)
    if num_views == 6:
        match_list = open(match_list_path, 'w')
        step_range = []

        for i in tqdm(range(0, num_images)):
            if i < num_per_view:
                step_range = list(range(1, min(7, num_per_view - i))) + list(
                    range(num_per_view, min(num_per_view + 7, 2 * num_per_view - i))) + list(
                    range(2 * num_per_view, min(2 * num_per_view + 7, 3 * num_per_view - i)))
            elif i >= num_per_view and i < 2 * num_per_view:
                step_range = list(range(1, min(7, 2 * num_per_view - i))) + list(
                    range(2 * num_per_view, min(2 * num_per_view + 7, 4 * num_per_view - i)))
            elif i >= 2 * num_per_view and i < 3 * num_per_view:
                step_range = list(range(1, min(7, 3 * num_per_view - i))) + list(
                    range(2 * num_per_view, min(2 * num_per_view + 7, 5 * num_per_view - i)))
            elif i >= 3 * num_per_view and i < 4 * num_per_view:
                step_range = list(range(1, min(7, 4 * num_per_view - i))) + list(
                    range(2 * num_per_view, min(2 * num_per_view + 7, 6 * num_per_view - i)))
            elif i >= 4 * num_per_view and i < 5 * num_per_view:
                step_range = list(range(1, min(7, 5 * num_per_view - i))) + list(
                    range(num_per_view, min(num_per_view + 7, 6 * num_per_view - i)))
            elif i >= 5 * num_per_view:
                step_range = list(range(1, min(7, 6 * num_per_view - i)))
            for step in step_range:
                if i + step >= num_images:
                    continue
                match_list.write("%s %s\n" % (folder_image_name_list[i], folder_image_name_list[i + step]))
                indices_flag = 3
                if 3 * num_per_view <= i < 4 * num_per_view and (i + step) >= 5 * num_per_view:  ####前左与前

                    K2_raw = sps[folder_image_name_list[i + step]]['keypoints']
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] < 960]
                    indices_flag = 1

                elif 4 * num_per_view <= i < 5 * num_per_view and (i + step) >= 5 * num_per_view:  ####前右与前

                    K2_raw = sps[folder_image_name_list[i + step]]['keypoints']
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] > 960]
                    indices_flag = 1

                elif i < num_per_view and num_per_view <= (i + step) < 2 * num_per_view:  ####后与后左

                    K1_raw = sps[folder_image_name_list[i]]['keypoints']
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K1_raw) if kp[0] > 960]
                    indices_flag = 0

                elif i < num_per_view and 2 * num_per_view <= (i + step) < 3 * num_per_view:  ####后与后右

                    K1_raw = sps[folder_image_name_list[i]]['keypoints']
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K1_raw) if kp[0] < 960]
                    indices_flag = 0

                else:
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices_flag = 3

                    # Check if descriptors are not empty before calling mutual_nn_matcher
                if D1.shape[0] > 0 and D2.shape[0] > 0:
                    if indices_flag == 1:
                        matches = mut_nn_matcher_indices_1(D1, D2, indices).astype(np.uint32)
                    elif indices_flag == 0:
                        matches = mut_nn_matcher_indices_0(D1, D2, indices).astype(np.uint32)
                    else:
                        matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
                    db.add_matches(i + 1, i + step + 1, matches)
                else:
                    print("Warning: Descriptors for images", images_name[i], "and", images_name[i + step], "are empty.")
        # print("ok3333333333333333")
        # exit()
        match_list.close()

    elif num_views == 3:
        print("3视角")
        match_list = open(match_list_path, 'w')
        step_range = []
        for i in tqdm(range(0, num_images)):
            if i < num_per_view:
                step_range = list(range(1, min(7, num_per_view - i))) + list(
                    range(2 * num_per_view, min(2 * num_per_view + 7, 3 * num_per_view - i)))
            elif num_per_view <= i < 2 * num_per_view:
                step_range = list(range(1, min(7, 2 * num_per_view - i))) + list(
                    range(num_per_view, min(num_per_view + 7, 3 * num_per_view - i)))
            elif i >= 2 * num_per_view:
                step_range = list(range(1, min(7, 3 * num_per_view - i)))
            for step in step_range:
                if i + step >= num_images:
                    continue
                match_list.write("%s %s\n" % (folder_image_name_list[i], folder_image_name_list[i + step]))
                indices_flag = 3
                if i < num_per_view and (i + step) >= 2 * num_per_view:  ####前左与前

                    K2_raw = sps[folder_image_name_list[i + step]]['keypoints']
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] < 960]
                    indices_flag = 1
                elif num_per_view <= i < 2 * num_per_view <= (i + step):  ####前右与前

                    K2_raw = sps[folder_image_name_list[i + step]]['keypoints']
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] > 960]
                    indices_flag = 1
                else:
                    D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                    D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                    indices_flag = 3
                    # Check if descriptors are not empty before calling mutual_nn_matcher
                if D1.shape[0] > 0 and D2.shape[0] > 0:
                    if indices_flag == 1:
                        matches = mut_nn_matcher_indices_1(D1, D2, indices).astype(np.uint32)
                    elif indices_flag == 0:
                        matches = mut_nn_matcher_indices_0(D1, D2, indices).astype(np.uint32)
                    else:
                        matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
                    db.add_matches(i + 1, i + step + 1, matches)
                else:
                    print("Warning: Descriptors for images", images_name[i], "and", images_name[i + step], "are empty.")
        match_list.close()
    elif num_views == 1:
        print("单视角")
        match_list = open(match_list_path, 'w')
        step_range = list(range(1, min(15, num_per_view)))
        for i in tqdm(range(0, num_images)):
            for step in step_range:
                if i + step >= num_images:
                    continue
                match_list.write("%s %s\n" % (folder_image_name_list[i], folder_image_name_list[i + step]))
                indices_flag = 3
                D1 = sps[folder_image_name_list[i]]['descriptors'] * 1.0
                D2 = sps[folder_image_name_list[i + step]]['descriptors'] * 1.0
                if D1.shape[0] > 0 and D2.shape[0] > 0:
                    matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
                    db.add_matches(i + 1, i + step + 1, matches)
                else:
                    print("Warning: Descriptors for images", images_name[i], "and", images_name[i + step], "are empty.")
    else:
        print("视角数不在1，3，6中")
        exit()


def match_glue(db, sps, images_name, match_list_path):
    print("match features by superglue with sequential match............................")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # sequential match
    step_range = [1, 2, 3, 5, 8, 13, 21, 44, 65, 109, 174, 210]
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')
    spg = SuperGlueMatching()
    total_iterations = len(step_range) * num_images - sum(step_range)  # 计算总迭代次数

    progress_bar = tqdm(total=total_iterations, unit='iteration')  # 创建进度条
    for step in step_range:
        for i in range(0, num_images - step):
            data = {}
            match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
            k0 = sps[images_name[i]]['keypoints']
            k1 = sps[images_name[i + step]]['keypoints']
            d0 = sps[images_name[i]]['descriptors']
            d1 = sps[images_name[i + step]]['descriptors']
            s0 = sps[images_name[i]]['scores']
            s1 = sps[images_name[i + step]]['scores']
            i0 = sps[images_name[i]]['image']
            i1 = sps[images_name[i + step]]['image']

            data['keypoints0'] = torch.from_numpy(k0).unsqueeze(0).to(device)
            data['keypoints1'] = torch.from_numpy(k1).unsqueeze(0).to(device)
            data['descriptors0'] = torch.from_numpy(d0.transpose()).unsqueeze(0).to(device)
            data['descriptors1'] = torch.from_numpy(d1.transpose()).unsqueeze(0).to(device)
            data['scores0'] = torch.from_numpy(s0.transpose()).unsqueeze(0).to(device)
            data['scores1'] = torch.from_numpy(s1.transpose()).unsqueeze(0).to(device)
            data['image0'] = i0
            data['image1'] = i1
            # print("111111111111111111111111")
            matches = get_super_glue_from_scenes_return(data, spg)

            # print("2222222222222222222222222222")
            # print("matches:", matches)
            # exit()
            db.add_matches(i + 1, i + step + 1, matches)
            progress_bar.update(1)  # 更新进度条

    match_list.close()
    # progress_bar.close()  # 关闭进度条


def geo_match(db, sps, images_name, folder, match_list_path, type):
    print("match features by geo match............................")
    if type == 0:
        match_list = get_match_list(folder)
    elif type == 1:
        match_list = merge_get_match_list(folder)
    else:
        match_list = turn_around_get_match_list(folder)
    #match_list = get_match_list(folder)
    num_images = len(match_list)
    print("Number of images:", num_images)
    match_list_txt = open(match_list_path, 'w')
    for i, match_images in enumerate(tqdm(match_list)):
        # print("image_id :", i + 1, "num of match_images:", len(match_images))
        for match_image in match_images:
            #print("match_image:", match_image)
            match_list_txt.write("%s %s\n" % (images_name[i], images_name[match_image]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[match_image]]['descriptors'] * 1.0
            matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
            #print('i+1',i+1)
            #print('match_image+1',match_image+1)
            #print(type(match_image))
            db.add_matches(i + 1, match_image + 1, matches)

    match_list_txt.close()


def compute_matrices(P1, P2, K1, K2):
    # 构建相机投影矩阵
    P = np.dot(P2, np.linalg.inv(P1))
    R = P[:3, :3]
    t = P[:3, 3]
    t_chapeau = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = np.dot(t_chapeau, R)
    F = np.dot(np.linalg.inv(K2).T, np.dot(E, np.linalg.inv(K1)))
    return F, E


def geo_verify(db):
    print("geo verify............................")
    # 读取db的matches表,读取每一列的数据,pair_id,rows,cols,data
    cursor = db.cursor()
    cursor.execute('SELECT camera_id, params FROM cameras')
    camera = {}
    for row in cursor.fetchall():
        camera_id, params = row
        params = blob_to_array(params, np.float64)
        K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
        camera[camera_id] = K
    cursor.execute(
        'SELECT image_id, prior_qw , prior_qx , prior_qy , prior_qz , prior_tx , prior_ty , prior_tz, camera_id  FROM images')
    image = {}
    for row in cursor.fetchall():
        image_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz, camera_id = row
        image[image_id] = {}
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat((prior_qx, prior_qy, prior_qz, prior_qw)).as_matrix()
        pose[:3, 3] = np.array([prior_tx, prior_ty, prior_tz])
        image[image_id]['pose'] = np.linalg.inv(pose)
        image[image_id]['K'] = camera[camera_id]

    cursor.execute('SELECT image_id, rows, cols,  data FROM keypoints')
    # 获取所有数据
    keypoint_data = cursor.fetchall()

    # 遍历数据并打印
    for row in keypoint_data:
        image_id, rows, cols, data = row
        image[image_id]['keypoints'] = blob_to_array(data, np.float32, (rows, cols))[:, :2]

    cursor.execute('SELECT pair_id,  rows, cols, data FROM matches')

    match_data = cursor.fetchall()
    for row in tqdm(match_data[10:]):
        pair_id, rows, cols, data = row
        # print("pair_id:", pair_id)
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        # print("image_id1:", image_id1, "image_id2:", image_id2)
        F, E = compute_matrices(image[image_id1]['pose'], image[image_id2]['pose'], image[image_id1]['K'],
                                image[image_id2]['K'])
        # print("F:", F, "E:", E)
        # 通过基础矩阵来验证匹配点对
        matches = blob_to_array(data, np.uint32, (rows, cols))
        to_delete = np.array([], dtype=bool)
        for i in range(matches.shape[0]):
            # for i in range(10):
            p1 = image[image_id1]['keypoints'][int(matches[i, 0])]
            p2 = image[image_id2]['keypoints'][int(matches[i, 1])]
            p1 = np.hstack((p1, 1))
            p2 = np.hstack((p2, 1))
            error = np.abs(np.dot(p2, np.dot(F, p1.T)))
            if error > 0.5:
                # 标记要删除的匹配点对
                to_delete = np.append(to_delete, True)
            else:
                to_delete = np.append(to_delete, False)
        # 使用布尔数组来删除匹配点对
        matches = np.delete(matches, to_delete, axis=0)
        db.add_two_view_geometry(image_id1, image_id2, matches, F, E)
    db.commit()
    cursor.close()

    '''cursor.execute('SELECT pair_id,  rows, cols FROM two_view_geometries')

    geo_match_data = cursor.fetchall()
    for row in tqdm(geo_match_data):
        pair_id, rows, cols = row
        print("pair_id:", pair_id)
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        print("image_id1:", image_id1, "image_id2:", image_id2)'''


# def match_features(db, sps, images_name, match_list_path):
#     print("match features by exhaustive match............................")
#     num_images = len(images_name)
#     match_list = open(match_list_path, 'w')
#
#     for i in range(num_images):
#         for j in range(i + 1, num_images):
#             match_list.write("%s %s\n" % (images_name[i], images_name[j]))
#             D1 = sps[images_name[i]]['descriptors'] * 1.0
#             D2 = sps[images_name[j]]['descriptors'] * 1.0
#             # Check if descriptors are not empty before calling mutual_nn_matcher
#             if D1.shape[0] > 0 and D2.shape[0] > 0:
#                 matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
#                 db.add_matches(i, j, matches)
#             else:
#                 print("Warning: Descriptors for images", images_name[i], "and", images_name[j], "are empty.")
#
#     match_list.close()


def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    duration = end - start
    print("[%s] cost %f s" % (cmd, duration))


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mapper(projpath, images_path):
    database_path = os.path.join(projpath, "database.db")
    colmap_sparse_path = os.path.join(projpath, "sparse")
    makedir(colmap_sparse_path)

    mapper = "colmap mapper --database_path %s --image_path %s --output_path %s" % (
        database_path, images_path, colmap_sparse_path
    )
    operate(mapper)


def geometric_verification(database_path, match_list_path):
    print("Running geometric verification..................................")
    cmd = "colmap matches_importer --database_path %s --match_list_path %s --match_type pairs" % (
        database_path, match_list_path
    )
    operate(cmd)


def superpoint_geomatch(projpath, sfm_folder="sfm", type=0):
    if type != 1:
        print(projpath)
        print(sfm_folder)
        sfm_path = os.path.join(projpath, sfm_folder, "chouzhen")
        print(sfm_path)
        if os.path.exists(os.path.join(projpath, "image")):
            image_path = os.path.join(projpath, "image")
        elif os.path.exists(os.path.join(projpath, "rawCamera")):
            image_path = os.path.join(projpath, "rawCamera")
        else:
            image_path = os.path.join(projpath, "rawData")
    else:
        sfm_path = os.path.join(projpath, sfm_folder)
        last_slash_index = projpath.rfind("/")
        print(last_slash_index)
        image_path = projpath[:last_slash_index]
        print("projpath:", projpath)

    database_path = os.path.join(sfm_path, "database.db")
    match_list_path = os.path.join(sfm_path, "image_pairs_to_match.txt")
    images_list_path = os.path.join(sfm_path, "images_list.txt")
    if os.path.exists(database_path):
        cmd = "rm -rf %s" % database_path
        operate(cmd)
    print(database_path)

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name = init_cameras_database(db, images_list_path, False)
    # 利用superpoint提取特征
    sps = import_feature(db, image_path, images_name)
    db.commit()
    db.close()  # 关闭数据库连接
    # 导入相机内参
    print("begin to import Camera parameters")
    camTodatabase(os.path.join(sfm_path, "cameras.txt"), database_path)
    print("Successfully imported camera parameters")

    db = COLMAPDatabase.connect(database_path)  # 重新打开数据库连接
    # 通过图像的位姿进行特征匹配
    geo_match(db, sps, images_name, sfm_path, match_list_path, type)
    print("match ok")
    # exit()

    db.commit()
    db.close()

    print("adding prior")
    # 添加先验信息
    add_all_prior(sfm_path)
    '''db = COLMAPDatabase.connect(database_path) # 重新打开数据库连接
    geo_verify(db)'''
    # 几何校验
    geometric_verification(database_path, match_list_path)

    # mapper(args.projpath, images_path)
    print("Feature matching and geometric verification completed successfully")


def superpoint_geomatch_clip(projpath, clip_id, chouzhen=True):
    if chouzhen:
        sfm_path = os.path.join(projpath, "sfm", "chouzhen", "clips", str(clip_id))
    else:
        sfm_path = os.path.join(projpath, "sfm")
    database_path = os.path.join(sfm_path, "database.db")
    print(database_path)
    match_list_path = os.path.join(sfm_path, "image_pairs_to_match.txt")
    if os.path.exists(database_path):
        cmd = "rm -rf %s" % database_path
        operate(cmd)
    images_list_path = os.path.join(sfm_path, "images_list.txt")
    if os.path.exists(os.path.join(projpath, "image")):
        image_path = os.path.join(projpath, "image")
    elif os.path.exists(os.path.join(projpath, "rawCamera")):
        image_path = os.path.join(projpath, "rawCamera")
    else:
        image_path = os.path.join(projpath, "rawData")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name = init_cameras_database(db, images_list_path, False)
    # print(images_name)
    sps = import_feature(db, image_path, images_name)
    db.commit()
    db.close()  # 关闭数据库连接
    # 导入相机内参
    print("begin to import Camera parameters")
    camTodatabase(os.path.join(sfm_path, "cameras.txt"), database_path)
    print("Successfully imported camera parameters")

    db = COLMAPDatabase.connect(database_path)  # 重新打开数据库连接
    geo_match(db, sps, images_name, sfm_path, match_list_path, chouzhen)
    print("match ok")
    # exit()

    db.commit()
    db.close()

    print("adding prior")
    add_all_prior(sfm_path)

    geometric_verification(database_path, match_list_path)

    # mapper(args.projpath, images_path)
    print("Feature matching and geometric verification completed successfully")


'''if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points colmap')
    parser.add_argument("--projpath", required=True, type=str)
    parser.add_argument("--chouzhen", required=False, type=bool, default=True)
    parser.add_argument("--single_camera", action='store_true')
    parser.add_argument("--matching_method", type=str, required=False, default="TraditionMatch",
                        choices=["MergeMatch", "TraditionMatch", "GeoMatch"])

    args = parser.parse_args()
    if args.chouzhen:
        sfm_path = os.path.join(args.projpath, "sfm", "chouzhen")
    else:
        sfm_path = os.path.join(args.projpath, "sfm")
    database_path = os.path.join(sfm_path, "database.db")
    match_list_path = os.path.join(sfm_path, "image_pairs_to_match.txt")
    if os.path.exists(database_path):
        cmd = "rm -rf %s" % database_path
        operate(cmd)
    images_list_path = os.path.join(sfm_path, "images_list.txt")
    image_path = os.path.join(args.projpath, "image")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name = init_cameras_database(db, images_list_path, args.single_camera)
    # print(images_name)
    sps = import_feature(db, image_path, images_name)
    db.commit()
    db.close()  # 关闭数据库连接
    # 导入相机内参
    print("begin to import Camera parameters")
    camTodatabase(os.path.join(sfm_path, "cameras.txt"), database_path)
    print("Successfully imported camera parameters")

    db = COLMAPDatabase.connect(database_path)  # 重新打开数据库连接
    # print("images_name:", images_name)
    # match_features(db, sps, images_name, match_list_path)
    if args.matching_method == "MergeMatch":
        print("begin to match images using MergeMatch")
        match_features_first(db, sps, images_name, match_list_path)
    elif args.matching_method == "TraditionMatch":
        print("begin to match images using TraditionMatch")
        match_features(db, sps, images_name, match_list_path)
    elif args.matching_method == "GeoMatch":
        print("begin to match images using GeoMatch")
        geo_match(db, sps, images_name, sfm_path, match_list_path)
    print("match ok")
    # exit()

    db.commit()
    db.close()

    print("adding prior")
    add_all_prior(sfm_path)

    geometric_verification(database_path, match_list_path)

    # mapper(args.projpath, images_path)
    print("Feature matching and geometric verification completed successfully")'''
if __name__ == '__main__':
    db = COLMAPDatabase.connect(
        '/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag/sfm/chouzhen/database.db')
    # geo_verify(db)
    superpoint_geomatch(
        '/dataset/rtfbag/merge_test1/ref_EP40-PVS-42_EP40_MDC_0430_0723_2023-09-27-08-03-00_0_merge_EP40-PVS-42_EP40_MDC_0430_0723_2023-09-27-08-05-02_4',
        False)
