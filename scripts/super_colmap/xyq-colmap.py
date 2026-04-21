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
from database import COLMAPDatabase
from transform_colmap_camera import camTodatabase
from add_prior import add_all_prior
import torch
from tqdm import tqdm

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


def init_cameras_database(db, images_path, cameratype, single_camera):
    print("init cameras database ......................................")
    images_name = []
    width = None
    height = None
    for name in sorted(os.listdir(images_path)):
        if 'jpg' in name or 'png' in name:
            images_name.append(name)
            if width is None:
                img = cv2.imread(os.path.join(images_path, name))
                height, width = img.shape[:2]
    cameraModel = camModelDict[cameratype]
    params = get_init_cameraparams(width, height, cameraModel)

    if single_camera:
        camera_id = 1
        db.add_camera(cameraModel, width, height, params, camera_id=camera_id)
        for i, name in enumerate(images_name):
            db.add_image(name, camera_id, image_id=i + 1)
    else:
        for i, name in enumerate(images_name):
            camera_id = i + 1
            db.add_camera(cameraModel, width, height, params, camera_id=camera_id)
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

def import_feature(db, images_path, images_name):
    print("feature extraction by super points ...........................")
    sps = get_super_points_from_scenes_return(images_path)
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
        "后缀   bin"
        image_name_gray = os.path.splitext(name)[0] + "bin.png"

        "后缀   _bin"
        # image_name_gray = os.path.splitext(name)[0] + ".png"

        image_path_gray = os.path.join(images_path, '..', 'seg', image_name_gray)
        gray_image = cv2.imread(image_path_gray, cv2.IMREAD_GRAYSCALE)

        # 使用布尔索引直接过滤关键点
        x = raw_keypoints[:, 0].astype(int)
        y = raw_keypoints[:, 1].astype(int)
        pixel_values = gray_image[y, x]
        valid_indices = np.logical_and(pixel_values != 0, ~np.isin(pixel_values,
                                                                   [1, 19, 20, 21, 22, 25, 26, 27, 28, 52, 53, 54, 55,
                                                                    56, 57, 58, 59, 60, 61, 62, 63, 64, 65]))

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
        ######################################
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
                                    np.ones((n_keypoints, 1)).astype(np.float32),
                                    np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i + 1, keypoints)
        # db.add_descriptors(i + 1, descriptors)
        keypoints_list.append((raw_keypoints, keypoints, name))
    ###################xyq
    # # 创建保存图片的文件夹
    # save_folder = '/data/sfm/xyq/test/1/sift_points'
    # os.makedirs(save_folder, exist_ok=True)
    #
    # for i, (raw_keypoints, keypoints, name) in enumerate(keypoints_list):
    #     image_path = os.path.join(images_path, name)
    #     image = cv2.imread(image_path)
    #
    #     # 绘制raw_keypoints在左图
    #     img_left = image.copy()
    #     for point in raw_keypoints:
    #         x, y = int(point[0]), int(point[1])
    #         cv2.circle(img_left, (x, y), 5, (0, 0, 255), -1)
    #
    #     # 绘制经过过滤的keypoints在右图
    #     img_right = image.copy()
    #     for point in keypoints:
    #         x, y = int(point[0]), int(point[1])
    #         cv2.circle(img_right, (x, y), 5, (0, 255, 0), -1)
    #
    #     # 将左图和右图拼接在一起
    #     combined_image = np.hstack((img_left, img_right))
    #
    #     # 构造保存图像的路径
    #     save_path = os.path.join(save_folder, f"Image_{i + 1}_with_Keypoints.jpg")
    #
    #     # 保存图像
    #     cv2.imwrite(save_path, combined_image)
    #
    #     # print(f"Image {i + 1} saved at {save_path}")
    #
    # print("All images saved successfully.")

    #########################

    return new_sps


def import_feature_from_sps(db, sps, images_name):
    print("feature extraction by super points ...........................")
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
                                    np.ones((n_keypoints, 1)).astype(np.float32),
                                    np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)


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

def match_features(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    # sequential match
    views = set()  # 用集合来保存不重复的视角
    for image_name in images_name:
        parts = image_name.split('_')  # 通过下划线将图像名称拆分成多个部分
        view = '_'.join(parts[:-1])  # 视角位于图像名称的除去最后一个部分之外的部分
        views.add(view)  # 将视角添加到集合中

    num_views = len(views)
    print("views:", views)
    print("Number of camera views:", num_views)

    num_images = len(images_name)
    num_per_view = num_images // num_views
    print("num_per_view:", num_per_view)
    if num_views == 6:
        match_list = open(match_list_path, 'w')
        step_range = []

        for i in tqdm(range(0, num_images)):
            if i < num_per_view:
                step_range = list(range(1, min(7, num_per_view-i))) + list(range(num_per_view, min(num_per_view+7, 2*num_per_view-i))) + list(range(2*num_per_view, min(2*num_per_view+7, 3*num_per_view-i)))
            elif i >= num_per_view and i < 2 * num_per_view:
                step_range = list(range(1, min(7, 2*num_per_view-i))) + list(range(2*num_per_view, min(2*num_per_view+7, 4*num_per_view-i)))
            elif i >= 2 * num_per_view and i < 3 * num_per_view:
                step_range = list(range(1, min(7, 3*num_per_view-i))) + list(range(2*num_per_view, min(2*num_per_view+7, 5*num_per_view-i)))
            elif i >= 3 * num_per_view and i < 4 * num_per_view:
                step_range = list(range(1, min(7, 4*num_per_view-i))) + list(range(2*num_per_view, min(2*num_per_view+7, 6*num_per_view-i)))
            elif i >= 4 * num_per_view and i < 5 * num_per_view:
                step_range = list(range(1, min(7, 5*num_per_view-i))) + list(range(num_per_view, min(num_per_view+7, 6*num_per_view-i)))
            elif i >= 5 * num_per_view:
                step_range = list(range(1, min(7, 6*num_per_view-i)))
            for step in step_range:
                if i + step >= num_images:
                    continue
                match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
                indices_flag = 3
                if 3 * num_per_view <= i < 4 * num_per_view and (i + step) >= 5 * num_per_view:  ####前左与前

                    K2_raw = sps[images_name[i+step]]['keypoints']
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] < 960]
                    indices_flag = 1

                elif 4 * num_per_view <= i < 5 * num_per_view and (i+step) >= 5 * num_per_view:  ####前右与前

                    K2_raw = sps[images_name[i+step]]['keypoints']
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] > 960]
                    indices_flag = 1

                elif i < num_per_view and num_per_view <= (i+step) < 2 * num_per_view:  ####后与后左

                    K1_raw = sps[images_name[i]]['keypoints']
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K1_raw) if kp[0] > 960]
                    indices_flag = 0

                elif i < num_per_view and 2 * num_per_view <= (i+step) < 3 * num_per_view:  ####后与后右

                    K1_raw = sps[images_name[i]]['keypoints']
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K1_raw) if kp[0] < 960]
                    indices_flag = 0

                else:
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
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
    elif num_views == 3:
        print("3视角")
        match_list = open(match_list_path, 'w')
        step_range = []
        for i in tqdm(range(0, num_images)):
            if i < num_per_view:
                step_range = list(range(1, min(7, num_per_view-i)))+ list(range(2 * num_per_view, min(2 * num_per_view + 7, 3 * num_per_view-i)))
            elif num_per_view <= i < 2 * num_per_view:
                step_range = list(range(1, min(7, 2*num_per_view-i))) + list(range(num_per_view, min(num_per_view + 7, 3 * num_per_view-i)))
            elif i >= 2 * num_per_view:
                step_range = list(range(1, min(7, 3*num_per_view-i)))
            for step in step_range:
                if i + step >= num_images:
                    continue
                match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
                indices_flag = 3
                if i < num_per_view and (i + step) >= 2 * num_per_view:  ####前左与前

                    K2_raw = sps[images_name[i + step]]['keypoints']
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] < 960]
                    indices_flag = 1
                elif num_per_view <= i < 2 * num_per_view <= (i + step):  ####前右与前

                    K2_raw = sps[images_name[i + step]]['keypoints']
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
                    indices = [idx for idx, kp in enumerate(K2_raw) if kp[0] > 960]
                    indices_flag = 1
                else:
                    D1 = sps[images_name[i]]['descriptors'] * 1.0
                    D2 = sps[images_name[i + step]]['descriptors'] * 1.0
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
                match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
                indices_flag = 3
                D1 = sps[images_name[i]]['descriptors'] * 1.0
                D2 = sps[images_name[i + step]]['descriptors'] * 1.0
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points colmap')
    parser.add_argument("--projpath", required=True, type=str)
    parser.add_argument("--cameraModel", type=str, required=False, default="SIMPLE_RADIAL")
    parser.add_argument("--images_path", required=False, type=str, default="rgb")
    parser.add_argument("--single_camera", action='store_true')
    parser.add_argument("--matching_method", type=str, required=False, default="TraditionMatch",
                        choices=["SuperGlue", "TraditionMatch"])

    args = parser.parse_args()
    database_path = os.path.join(args.projpath, "database.db")
    match_list_path = os.path.join(args.projpath, "image_pairs_to_match.txt")
    if os.path.exists(database_path):
        cmd = "rm -rf %s" % database_path
        operate(cmd)
    images_path = os.path.join(args.projpath, args.images_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name = init_cameras_database(db, images_path, args.cameraModel, args.single_camera)
    sps = import_feature(db, images_path, images_name)
    db.commit()
    db.close()  # 关闭数据库连接
    # 导入相机内参
    print("begin to import Camera parameters")
    camTodatabase(os.path.join(args.projpath, "cameras.txt"), database_path)
    print("Successfully imported camera parameters")

    db = COLMAPDatabase.connect(database_path)  # 重新打开数据库连接
    # print("images_name:", images_name)
    # match_features(db, sps, images_name, match_list_path)
    if args.matching_method == "SuperGlue":
        print("begin to match images using SuperGlue")
        match_glue(db, sps, images_name, match_list_path)
    elif args.matching_method == "TraditionMatch":
        print("begin to match images using TraditionMatch")
        match_features(db, sps, images_name, match_list_path)
    print("match ok")
    # exit()

    db.commit()
    db.close()

    print("adding prior")
    add_all_prior(args.projpath)

    geometric_verification(database_path, match_list_path)

    # mapper(args.projpath, images_path)
    print("Feature matching and geometric verification completed successfully")