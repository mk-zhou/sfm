from superpoint import SuperPoint
from superglue import SuperGlue
import cv2
import numpy as np
import torch
import json
import argparse
from tqdm import tqdm
import os

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": "superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        print("SuperPoint detector config: ")
        print(self.config)

        self.device = 'cuda:0' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        print("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # print("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})
        # print("pred", pred)
        ret_dict = {
            "image_size": [image.shape[0], image.shape[1]],
            # "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose(),
            # "keypoints_tensor": pred["keypoints"],
            # "scores_tensor": pred["scores"][0],
            # "descriptors_tensor": pred["descriptors"]
        }
        # print("ret_dict", ret_dict)
        # exit()
        return ret_dict


class SuperGlueMatching(object):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20,  ####raw 100
        'match_threshold': 0.01,  ####raw 0.2
        "path": "superglue_outdoor.pth",
        "cuda": True
    }

    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, config={}):
        super().__init__()
        self.config = self.default_config
        self.config = {**self.config, **config}
        print("SuperGlue matching config: ")
        print(self.config)
        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'
        print("creating SuperGlue matching...")
        # self.superpoint = SuperPoint(self.config).to(self.device)
        self.superglue = SuperGlue(self.config).to(self.device)

    def __call__(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        # print("data", data)

        pred = self.superglue(data)
        return pred


def get_super_points_from_scenes(image_path, result_dir):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    spd = SuperPointDetector()
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        ret_dict = spd(cv2.imread(image_name))
        with open(os.path.join(result_dir, name + ".json"), 'w') as f:
            json.dump(ret_dict, f)


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def get_super_points_from_scenes_return_rawwww(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    spd = SuperPointDetector()
    sps = {}
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        image = cv2.imread(image_name)
        ret_dict = spd(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_tensor = frame2tensor(image_gray, device)
        sps[name] = ret_dict
        sps[name]['image'] = frame_tensor
    return sps


def get_super_points_from_scenes_return(image_path, images_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    spd = SuperPointDetector()
    sps = {}
    for name in tqdm(sorted(images_name)):
        image_name = os.path.join(image_path, name)
        image = cv2.imread(image_name)
        if image.shape != (1920, 1080):
            image = cv2.resize(image, (1920, 1080))
            #print("image_name", image_name)
        # exit()
        ret_dict = spd(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_tensor = frame2tensor(image_gray, 'cpu')
        # folder_image_name = f"{name[0]}_{name[1]}"
        # print("folder_image_name", folder_image_name)
        # exit()
        sps[name] = ret_dict
        sps[name]['image'] = frame_tensor
    # print("sps", sps)
    # exit()
    return sps


def get_super_glue_from_scenes_return(data, spg):
    # spg = SuperGlueMatching()
    pred = spg(data)
    matches_raw = pred['matches0'][0].cpu().numpy()
    # 使用mutual_nn_matcher的输出格式
    matches = []
    matched_indices = np.where(matches_raw != -1)[0]
    for idx in matched_indices:
        matches.append([idx + 1, matches_raw[idx]])

    matches = np.array(matches, dtype=np.uint32)

    # print("kkk", data['keypoints0'].shape)
    # print("matches", matches)
    # print("matches.shape", matches.shape)
    # exit()
    return matches


def get_sift_features_from_scenes(image_path):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    sift = cv2.SIFT_create()
    sift_features = {}
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(image, None)

        # Convert keypoints to a NumPy array
        keypoints_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # Convert descriptors to NumPy array and transpose
        descriptors_np = descriptors if descriptors is not None else np.empty((0, 128), dtype=np.float32)

        sift_features[name] = {"keypoints": keypoints_np, "descriptors": descriptors_np}
    return sift_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points detector')
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=False, default="../superpoints",
                        help="real result_file = args.image_path + args.result_dir")
    args = parser.parse_args()
    result_dir = os.path.join(args.image_path, args.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    get_super_points_from_scenes(args.image_path, result_dir)
