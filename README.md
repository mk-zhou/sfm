<div align="center">

# [ICRA'26] MRASfM: Multi-Camera Reconstruction and Aggregation through Structure-from-Motion in Driving Scenes
[Lingfeng Xuan†, Chang Nie†, Yiqing Xu, Yanzi Miao, and Hesheng Wang^]

</div>


<div align="center">
  <h2 align="center">
  <a href="https://arxiv.org/abs/2510.15467" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.09456-b31b1b.svg?style=flat">
  </a>
</div>



> **Abstract:** Structure from Motion (SfM) estimates camera poses and reconstructs point clouds, forming a foundation for various tasks. However, applying SfM to driving scenes captured by multi-camera systems presents significant difficulties, including unreliable pose estimation, excessive outliers in road surface reconstruction, and low reconstruction efficiency. To address these limitations, we propose a Multi-camera Reconstruction and Aggregation Structure-from-Motion (MRASfM) framework specifically designed for driving scenes. MRASfM enhances the reliability of camera pose estimation by leveraging the fixed spatial relationships within the multi-camera system during the registration process. To improve the quality of road surface reconstruction, our framework employs a plane model to effectively remove erroneous points from the triangulated road surface. Moreover, treating the multi-camera set as a single unit in Bundle Adjustment (BA) helps reduce optimization variables to boost efficiency. In addition, MRASfM achieves multi-scene aggregation through scene association and assembly modules in a coarse-to-fine fashion. We deployed multi-camera systems on actual vehicles to validate the generalizability of MRASfM across various scenes and its robustness in challenging conditions through real-world applications. Furthermore, large-scale validation results on public datasets show the state-of-the-art performance of MRASfM, achieving 0.124 absolute pose error on the nuScenes dataset. The code is available at https://github.com/IRMVLab/MRASfM.

---
### Installation
1. Clone this repository:
```bash
git clone https://github.com/IRMVLab/MRASfM.git
```

2. Create the conda environment:
```bash
conda create -n mrasfm python=3.10 -y
conda activate mrasfm
pip install --upgrade pip  
pip install -r requirements.txt    
```

3. Install MRASfM

Install COLMAP dependencies** using the [COLMAP build guide](https://colmap.github.io/install.html#build-from-source).
```
cd MRASfM/colmap
mkdir build && cd build
cmake .. -GNinja
ninja && sudo ninja install
```
---
### Usage
1. KITTI odometry
```
python make_odometry_input.py --yaml_path <odometry_sfm.yaml> 
```

2. NuScenes 
```
python make_nuscene_input.py --yaml_path <nuscene_sfm.yaml> 
```

3. Scene Merge

```
python make_merge_input.py --yaml_path <merge_sfm.yaml> 
```

---

### Citation
```ruby
@misc{xuan2025mrasfmmulticamerareconstructionaggregation,
      title={MRASfM: Multi-Camera Reconstruction and Aggregation through Structure-from-Motion in Driving Scenes}, 
      author={Lingfeng Xuan and Chang Nie and Yiqing Xu and Zhe Liu and Yanzi Miao and Hesheng Wang},
      year={2025},
      eprint={2510.15467},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.15467}, 
}
```
