import subprocess
import os
from os.path import join
from utils.sigma_remove_noise import get_sigma_road_ply
from utils.road_sparse_rgb import get_road_ply
from utils.rt import get_rt_sigma_road_ply
from utils.seg_road import get_seg_road_folder
from utils.colmap2mvsnet_acm import my_processing_single_scene
from utils.seg_road_auto import get_seg_road_folder_auto
from utils.rm_road_noise import rm_road_noise
from utils.remove_ransac_noise_dense_auto import rm_ransac_points
from utils.rm_noise_trajectory import rm_tra_points
from utils.uniform_height_auto import grid_uni_Z
from utils.seg_dynamic_auto import get_seg_dynamic_folder
from utils.remove_noise_points_auto import remove_noise_points
from utils.dmb2bin import process_ACMP
from utils.creat_dense_ACMP import process_dense_ACMP_multiprocess

from PIL import Image
from tqdm import tqdm
import shutil


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


# 对场景做相机组重建
def rig_mapper(scene):
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    sfm_path = join(scene, 'sfm', 'chouzhen')
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, 'rig_mapper')
    rig_json_path = join(sfm_path, 'rig.json')
    mapper_str = "colmap mapper --database_path " + database_path + ' --image_path ' + image_path + ' --output_path ' + output_path + ' --rig_config_path ' + rig_json_path
    mkdir_output_str = "mkdir " + output_path
    os.system(mkdir_output_str)
    os.system(mapper_str)


def rig_mapper_new(scene,out_folder):
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    sfm_path = join(scene, 'sfm', 'chouzhen')
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, out_folder)
    rig_json_path = join(sfm_path, 'rig.json')
    mapper_str = "colmap mapper --database_path " + database_path + ' --image_path ' + image_path + ' --output_path ' + output_path + ' --rig_config_path ' + rig_json_path
    mkdir_output_str = "mkdir " + output_path
    os.system(mkdir_output_str)
    os.system(mapper_str)

#colmap原生重建
def mapper(scene):
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    sfm_path = join(scene, 'sfm', 'chouzhen')
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, 'rig_mapper')
    rig_json_path = join(sfm_path, 'rig.json')
    mapper_str = "colmap mapper --database_path " + database_path + ' --image_path ' + image_path + ' --output_path ' + output_path
    mkdir_output_str = "mkdir " + output_path
    os.system(mkdir_output_str)
    os.system(mapper_str)

# 将sfm的结果转换为txt格式
def get_txt_from_scene(scene):
    rig_mapper_path = join(scene, 'sfm', 'chouzhen', 'rig_mapper')
    txt_path = join(rig_mapper_path, 'txt')
    if not os.path.exists(txt_path, 'points3D.txt'):
        bin_path = join(rig_mapper_path, '0')
        mkdir_txt_str = "mkdir " + txt_path
        bin2txt_str = "colmap model_converter --output_type TXT --input_path " + bin_path + " --output_path " + txt_path
        os.system(mkdir_txt_str)
        os.system(bin2txt_str)
    else:
        print('txt exists')


# 得到场景的稀疏路面点
def get_road_points_from_scene(scene):
    if not os.path.exists(join(scene, 'sfm', 'chouzhen', 'rig_mapper', 'txt', 'points3D.txt')):
        print('points3D.txt not exists')
        get_txt_from_scene(scene)
    get_seg_road_folder(scene)
    get_road_ply(scene)
    get_sigma_road_ply(scene)
    get_rt_sigma_road_ply(scene)


# 对场景进行稠密重建
def get_dense_ply_from_scene(scene):
    input_path = join(scene, 'sfm', 'chouzhen', 'rig_mapper', '0')
    dense_path = join(scene, 'sfm', 'chouzhen', 'dense')
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
        fw_folder_path = join(dense_path, 'images/camera-73-encoder-undistort')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')
    else:
        image_path = join(scene, 'rawData')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')

    undistorter_str = ('colmap image_undistorter --image_path ' + image_path + ' --input_path ' + input_path +
                       ' --output_path ' + dense_path)
    patch_match_str = 'colmap patch_match_stereo --workspace_path ' + dense_path + (' --PatchMatchStereo'
                                                                                    '.geom_consistency 0 '
                                                                                    '--PatchMatchStereo.depth_max 150 --PatchMatchStereo.depth_min 0.001')
    stereo_fusion_str = 'colmap stereo_fusion --input_type photometric --workspace_path ' + dense_path + ' --output_path ' + join(
        dense_path,
        'dense.ply')
    os.system(undistorter_str)
    resize_folder(fw_folder_path)
    os.system(patch_match_str)
    os.system(stereo_fusion_str)
    all_dense_fusion_rmdynamic(scene, sfm_folder='sfm', dense_folder='dense')
    all_get_road_points(scene, sfm_folder='sfm', dense_folder='dense')
    all_road_points_process(scene, sfm_folder='sfm', dense_folder='dense')


# 对场景做直接三角化重建
def point_triangulator_from_scene(scene):
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    sfm_path = join(scene, 'fn_prior_sfm', 'chouzhen')
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, 'tri')

    mkdir_output_str = "mkdir " + output_path
    tri_str = "colmap point_triangulator --database_path " + database_path + ' --image_path ' + image_path + ' --input_path ' + sfm_path + ' --output_path ' + output_path
    os.system(mkdir_output_str)
    os.system(tri_str)


def point_triangulator_from_merge_scene(scene):
    last_slash_index = scene.rfind("/")
    image_path = scene[:last_slash_index]
    sfm_path = join(scene, 'sfm')
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, 'tri')

    mkdir_output_str = "mkdir " + output_path
    tri_str = "colmap point_triangulator --database_path " + database_path + ' --image_path ' + image_path + ' --input_path ' + sfm_path + ' --output_path ' + output_path
    os.system(mkdir_output_str)
    os.system(tri_str)


def merge_mapper(scene):
    last_slash_index = scene.rfind("/")
    image_path = scene[:last_slash_index]
    sfm_path = join(scene, 'sfm')
    database_path = join(sfm_path, 'database.db')
    mapper_path = join(sfm_path, 'merge_mapper')
    rig_json_path = join(sfm_path, 'rig.json')
    if not os.path.exists(mapper_path):
        os.makedirs(mapper_path)
    mapper_str = ("colmap mapper --database_path " + database_path + ' --image_path '
                  + image_path + ' --output_path ' + mapper_path + ' --rig_config_path '
                  + rig_json_path + ' --ref_images_list_path ' + sfm_path + '/ref_images_list.txt ')
    print(mapper_str)
    os.system(mapper_str)


def point_triangulator_from_scene_clip(scene, clip_id):
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    sfm_path = join(scene, 'sfm', 'chouzhen', 'clips', str(clip_id))
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, 'tri')

    mkdir_output_str = "mkdir " + output_path
    tri_str = "colmap point_triangulator --database_path " + database_path + ' --image_path ' + image_path + ' --input_path ' + sfm_path + ' --output_path ' + output_path
    os.system(mkdir_output_str)
    os.system(tri_str)


# 对三角化的结果做稠密重建
def tri_get_dense_ply_from_scene(scene):
    input_path = join(scene, 'sfm', 'chouzhen', 'tri')
    dense_path = join(scene, 'sfm', 'chouzhen', 'dense_tri')
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
        fw_folder_path = join(dense_path, 'images/camera-73-encoder-undistort')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')
    else:
        image_path = join(scene, 'rawData')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')

    undistorter_str = ('colmap image_undistorter --image_path ' + image_path + ' --input_path ' + input_path +
                       ' --output_path ' + dense_path)
    patch_match_str = 'colmap patch_match_stereo --workspace_path ' + dense_path
    stereo_fusion_str = 'colmap stereo_fusion --workspace_path ' + dense_path + ' --output_path ' + join(dense_path,
                                                                                                         'dense.ply')
    os.system(undistorter_str)
    resize_folder(fw_folder_path)
    os.system(patch_match_str)
    os.system(stereo_fusion_str)


def get_mapper_result_and_tri(scene, type=''):
    sfm_path = join(scene, type + 'sfm')
    txt_folder = join(sfm_path, 'chouzhen', 'rig_mapper', 'txt')
    work_path = join(sfm_path, 'chouzhen', 'mapper_as_input')
    if not os.path.exists(txt_folder):
        return
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    input_images_txt_path = join(txt_folder, 'images.txt')
    output_images_txt_path = join(work_path, 'images.txt')
    with open(input_images_txt_path, 'r') as f:
        lines = f.readlines()[4::2]
    with open(output_images_txt_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    cameras_txt_path = join(txt_folder, 'cameras.txt')
    output_cameras_txt_path = join(work_path, 'cameras.txt')
    shutil.copy(cameras_txt_path, output_cameras_txt_path)
    output_points3D_txt_path = join(work_path, 'points3D.txt')
    dense_path = join(work_path, 'dense_tri')
    with open(output_points3D_txt_path, 'w') as file:
        pass
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
        fw_folder_path = join(dense_path, 'images/camera-73-encoder-undistort')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')
    else:
        image_path = join(scene, 'rawData')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')
    input_path = join(scene, 'sfm', 'chouzhen', 'mapper_as_input')
    database_path = join(scene, 'sfm', 'chouzhen', 'or_database.db')
    output_path = join(work_path, 'mapper_as_input_tri')
    mkdir_output_str = "mkdir " + output_path
    tri_str = "colmap point_triangulator --database_path " + database_path + ' --image_path ' + image_path + ' --input_path ' + input_path + ' --output_path ' + output_path
    os.system(mkdir_output_str)
    os.system(tri_str)
    undistorter_str = ('colmap image_undistorter --image_path ' + image_path + ' --input_path ' + output_path +
                       ' --output_path ' + dense_path)
    patch_match_str = 'colmap patch_match_stereo --workspace_path ' + dense_path
    stereo_fusion_str = 'colmap stereo_fusion --workspace_path ' + dense_path + ' --output_path ' + join(dense_path,
                                                                                                         'dense.ply')
    os.system(undistorter_str)
    resize_folder(fw_folder_path)
    os.system(patch_match_str)
    os.system(stereo_fusion_str)


def rig_mapper_clip(scene, clip_id):
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    sfm_path = join(scene, 'sfm', 'chouzhen', 'clips', str(clip_id))
    database_path = join(sfm_path, 'database.db')
    output_path = join(sfm_path, 'rig_mapper')
    rig_json_path = join(sfm_path, 'rig.json')
    mapper_str = "colmap mapper --database_path " + database_path + ' --image_path ' + image_path + ' --output_path ' + output_path + ' --rig_config_path ' + rig_json_path
    mkdir_output_str = "mkdir " + output_path
    os.system(mkdir_output_str)
    os.system(mapper_str)


def get_dense_ply_from_scene_clip(scene, clip_id):
    input_path = join(scene, 'sfm', 'chouzhen', 'clips', str(clip_id), 'rig_mapper', '0')
    dense_path = join(scene, 'sfm', 'chouzhen', 'clips', str(clip_id), 'dense')
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
        fw_folder_path = join(dense_path, 'images/camera-73-encoder-undistort')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')
    else:
        image_path = join(scene, 'rawData')
        fw_folder_path = join(dense_path, 'images/camera-73-undistort')
    # image_path = join(scene, 'seg')
    undistorter_str = ('colmap image_undistorter --image_path ' + image_path + ' --input_path ' + input_path +
                       ' --output_path ' + dense_path)
    patch_match_str = 'colmap patch_match_stereo --workspace_path ' + dense_path
    stereo_fusion_str = 'colmap stereo_fusion --workspace_path ' + dense_path + ' --output_path ' + join(dense_path,
                                                                                                         'dense.ply')
    os.system(undistorter_str)
    resize_folder(fw_folder_path)
    '''os.system(patch_match_str)
    os.system(stereo_fusion_str)'''


# 保留地面mask，地面融合
def all_get_road_points(scene, sfm_folder='sfm', dense_folder='dense'):
    seg_folder = join(scene, 'seg')
    output_folder = join(scene, sfm_folder, 'chouzhen', 'seg_road_resize')
    image_list_txt = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')

    workspace_path = join(scene, sfm_folder, 'chouzhen', dense_folder)
    output_path = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_raw.ply')
    mask_path = join(scene, sfm_folder, 'chouzhen', 'seg_road')
    dense_colmap_road_str = "colmap stereo_fusion --input_type photometric --workspace_path " + workspace_path + " --output_path " + output_path + " --StereoFusion.mask_path " + mask_path

    # get_seg_road_folder_auto(seg_folder, output_folder, image_list_txt)
    os.system(dense_colmap_road_str)


# 地面点处理：半径去噪，ransac，轨迹点过滤，拍扁
def all_road_points_process(scene, sfm_folder='sfm', dense_folder='dense'):
    dense_ACMP_folder = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP')

    input_r = join(scene, sfm_folder, 'chouzhen', dense_ACMP_folder, 'acmap_road.ply')
    output_r = input_ransac = join(scene, sfm_folder, 'chouzhen', dense_ACMP_folder, 'acmap_road_fil.ply')
    output_ransac = input_road = join(scene, sfm_folder, 'chouzhen', dense_ACMP_folder, 'acmap_road_ransac.ply')
    input_carply = join(scene, sfm_folder, 'car.ply')
    output_road = input_uniz = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_rm10.ply')
    output_uniz = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_road_rm10_uniZ.ply')

    rm_road_noise(input_r, output_r)
    rm_ransac_points(input_ransac, output_ransac)
    # rm_tra_points(input_road, input_carply, output_road, radius=10)
    # grid_uni_Z(input_uniz, output_uniz, grid_size=2)


# 去动态物mask生成，去除动态物融合,去噪
def all_dense_fusion_rmdynamic(scene, sfm_folder='sfm', dense_folder='dense'):
    seg_folder = join(scene, 'seg')
    output_folder = join(scene, sfm_folder, 'chouzhen', 'seg_dynamic')
    image_list_txt = join(scene, sfm_folder, 'chouzhen', 'images_list.txt')
    workspace_path = join(scene, sfm_folder, 'chouzhen', dense_folder)
    output_path = input_rm = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_rmdy.ply')
    mask_path = join(scene, sfm_folder, 'chouzhen', 'seg_dynamic')
    output_rm = join(scene, sfm_folder, 'chouzhen', dense_folder, 'dense_rmdy_fil.ply')
    dense_colmap_road_str = "colmap stereo_fusion --input_type photometric --workspace_path " + workspace_path + " --output_path " + output_path + " --StereoFusion.mask_path " + mask_path

    get_seg_dynamic_folder(seg_folder, output_folder, image_list_txt)
    os.system(dense_colmap_road_str)
    remove_noise_points(input_rm, output_rm)

def judgement_fw_folder_path(scene, dense_folder ):
    if os.path.exists(join(scene, 'image')):
        dense_image_path = join(dense_folder, 'images')
        if os.path.exists(join(dense_image_path, 'camera-0-undistort')):
            fw_folder_path = join(dense_image_path, 'camera-0-undistort')
        elif os.path.exists(join(dense_image_path, 'camera-1-undistort')):
            fw_folder_path = join(dense_image_path, 'camera-1-undistort')
        elif os.path.exists(join(dense_image_path, 'soc_encoded_camera_0-undistort')):
            fw_folder_path = join(dense_image_path, 'soc_encoded_camera_0-undistort')
        else:
            fw_folder_path = join(dense_image_path, 'soc_encoded_camera_1-undistort')
    elif os.path.exists(join(scene, 'rawCamera')):
        fw_folder_path = join(dense_folder, 'images/camera-73-undistort')
    else:
        fw_folder_path = join(dense_folder, 'images/camera-73-undistort')

    return fw_folder_path

# 降分辨率,ACMP深度估计，colmap融合，地面点云去噪
def dense_ACMP_colmap(scene, sfm_folder='sfm', dense_folder='dense', yaml_path = 'orin_sfm.yaml'):
    rig_mapper_path = join(scene, sfm_folder, 'chouzhen', 'rig_mapper', '0')
    dense_folder = join(scene, sfm_folder, 'chouzhen', dense_folder)
    if os.path.exists(join(scene, 'image')):
        image_path = join(scene, 'image')
    elif os.path.exists(join(scene, 'rawCamera')):
        image_path = join(scene, 'rawCamera')
    else:
        image_path = join(scene, 'rawData')
    save_folder = join(scene, sfm_folder, 'chouzhen')
    acmp_path = "./ACMP-t4/ACMP"  # 定义 ACMP 可执行文件的路径
    dense_ACMP_folder = join(scene, sfm_folder, 'chouzhen', 'dense_ACMP')
    acmap_road_path = join(dense_ACMP_folder, 'acmap_road.ply')
    mask_path = join(scene, sfm_folder, 'chouzhen', 'seg_road')
    yaml_path = os.path.abspath(yaml_path)

    undistorter_str = (
            'colmap image_undistorter --image_path ' + image_path + ' --input_path ' + rig_mapper_path + ' --output_path ' + dense_folder)
    #colmap2acm_str = "python /dataset/sfm/scripts/ACMP-t4/colmap2mvsnet_acm.py --resize 1 --dense_folder " + dense_folder + " --save_folder " + save_folder
    acmp_cmd = acmp_path + " " + save_folder + " " + yaml_path# 使用完整路径拼接 ACMP 命令
    dense_colmap_road_str = "colmap stereo_fusion --input_type photometric --workspace_path " + dense_ACMP_folder + " --output_path " + acmap_road_path + " --StereoFusion.mask_path " + mask_path

    #colmap去畸变并生成dense文件夹
    os.system(undistorter_str)
    # 前视相机resize
    fw_folder_path = judgement_fw_folder_path(scene, dense_folder)
    resize_folder(fw_folder_path)
    # 生成路面掩膜
    get_seg_road_folder_auto(scene, sfm_folder, dense_folder)
    # colmap位姿数据转换为ACMP格式
    my_processing_single_scene(dense_folder, save_folder)
    print("colmap2acm done")
    # ACMP深度估计
    subprocess.run(acmp_cmd, shell=True)
    print("acmp done")
    # ACMP深度dmb文件转为colmap融合需要的bin文件格式
    process_ACMP(scene, sfm_folder)
    # 复制colmap融合需要文件，并删除重复及不需要的文件
    process_dense_ACMP_multiprocess(scene, sfm_folder, dense_folder)
    # 融合点云
    os.system(dense_colmap_road_str)
    print("dense_colmap_road_str done")
    # 地面点云去噪
    all_road_points_process(scene, sfm_folder, dense_folder)
    # print("all_road_points_process done")


if __name__ == '__main__':
    '''scenes = ['/nas_gt/awesome_gt/Lane_GT_GE/appen_data/20230905/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-28-18-24-19_52.bag',
              '/nas_gt/awesome_gt/Lane_GT_GE/appen_data/20230905/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-28-18-26-20_56.bag']
    for scene in scenes:
        #get_dense_ply_from_scene(scene)
        point_triangulator_from_scene(scene)
        tri_get_dense_ply_from_scene(scene)'''
    #dense_ACMP_colmap('/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/0618/EP41-ORIN-0S13-00G_FULL_MCAP_02.22.03_20240415-041407_3.mcap')
    rig_mapper('/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/0618/EP41-ORIN-0S13-00G_FULL_MCAP_02.22.03_20240415-041207_1.mcap')
    # point_triangulator_from_scene('/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/0618/EP41-ORIN-0S13-00G_FULL_MCAP_02.22.03_20240415-041207_1.mcap')
    # get_dense_ply_from_scene_clip('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag',0)
    # all_dense_fusion_rmdynamic('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag')
    # rig_mapper_clip('/dataset/ORIN_DATA/ORIN01/20231101-033938_38', 0)
    # get_dense_ply_from_scene_clip('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-31_43.bag', 1)
    #mapper('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-31_43.bag')
    #point_triangulator_from_scene('/dataset/sfm/dataset/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.09_1225_20231101-025107_39.mcap')
    # point_triangulator_from_scene('/dataset/maptr_data/appen_data/20230831/EP40-PVS-42_EP40_MDC_0930_1215_2023-07-27-13-48-01_42.bag')
    #point_triangulator_from_merge_scene('/dataset/rtfbag/merge_test2/ref_EP40-PVS-42_EP40_MDC_0430_0723_2023-09-27-08-12-54_1_merge_EP40-PVS-42_EP40_MDC_0430_0723_2023-09-27-08-15-52_7')
    # merge_mapper('/dataset/rtfbag/merge_datasets/557039854_190/has_pgo/ref_EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-022013_15_merge_EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-022113_16')
    #dense_ACMP_colmap('/dataset/rtfbag/merge_test2/EP40-PVS-42_EP40_MDC_0430_0723_2023-09-27-08-52-55_3')
    #point_triangulator_from_merge_scene('/dataset/rtfbag/merge_datasets/557039854_190/has_pgo/ref_EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-022013_15_merge_EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-022113_16')
    # point_triangulator_from_scene('/dataset/rtfbag/merge_datasets/557039854_464/EP41-ORIN-0S14-00G_EP41_ORIN_02.10.12_1225_20231101-001301_10')
    #all_road_points_process('/dataset/handsome_man_with_handsome_data/sfm_test/input/1', 'sfm', 'dense')

    #dense_ACMP_colmap('/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/data/EP41-ORIN-0S14-00G_20240523_081429/EP41-ORIN-0S14-00G_FULL_MCAP_02.22.03_20240414-105625_53.mcap')

    '''folder = '/vepfs_dataset/handsome_man_with_handsome_data/sfm_debug/0618'
    sub_folders = sorted(os.listdir(folder))
    for sub_folder in sub_folders[4:]:
        folder_path = os.path.join(folder, sub_folder)
        dense_ACMP_colmap(folder_path, 'sfm', 'dense')'''