// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "base/camera_rig.h"

#include "util/misc.h"

namespace colmap {

    CameraRig::CameraRig() {}

    size_t CameraRig::NumCameras() const { return rig_cameras_.size(); }

    size_t CameraRig::NumSnapshots() const { return snapshots_.size(); }

    bool CameraRig::HasCamera(const camera_t camera_id) const {
        return rig_cameras_.count(camera_id);
    }

    camera_t CameraRig::RefCameraId() const { return ref_camera_id_; }

    void CameraRig::SetRefCameraId(const camera_t camera_id) {
        CHECK(HasCamera(camera_id));
        ref_camera_id_ = camera_id;
    }

    // 获取相机系统中的所有相机ID
    std::vector<camera_t> CameraRig::GetCameraIds() const {
        std::vector<camera_t> rig_camera_ids;          // 创建存储相机ID的向量
        rig_camera_ids.reserve(NumCameras());          // 预留足够的空间以容纳所有相机ID
        for (const auto &rig_camera: rig_cameras_) {   // 遍历相机系统中的每个相机
            rig_camera_ids.push_back(rig_camera.first);// 将相机ID添加到相机ID向量中
        }
        return rig_camera_ids;// 返回相机ID向量
    }

    //得到image_id所在的snapshot编号
    size_t CameraRig::GetSnapshotId(image_t image_id) const {

        for (size_t snapshot_idx = 0; snapshot_idx < snapshots_.size(); ++snapshot_idx) {
            const auto &snapshot = snapshots_[snapshot_idx];
            if (std::find(snapshot.begin(), snapshot.end(), image_id) != snapshot.end()) {
                return snapshot_idx;// 返回包含图像 ID 的快照编号
            }
        }
        return -1;// 如果未找到包含图像 ID 的快照，返回 -1 表示未找到
    }


    //得到image_id所在的snapshot里面所有的图片编号
    std::vector<image_t> CameraRig::GetImageId(size_t snapshot_id) const {
        // 检查快照编号是否有效
        if (snapshot_id >= snapshots_.size()) {
            // 返回一个空的图像 ID 向量表示无效的快照编号
            return std::vector<image_t>();
        }

        // 返回指定快照中的图像 ID 向量
        return snapshots_[snapshot_id];
    }

    const std::vector<std::vector<image_t>> &CameraRig::Snapshots() const {
        return snapshots_;
    }

    void CameraRig::AddCamera(const camera_t camera_id,
                              const Eigen::Vector4d &rel_qvec,
                              const Eigen::Vector3d &rel_tvec) {
        CHECK(!HasCamera(camera_id));
        CHECK_EQ(NumSnapshots(), 0);
        RigCamera rig_camera;
        rig_camera.rel_qvec = rel_qvec;
        rig_camera.rel_tvec = rel_tvec;
        rig_cameras_.emplace(camera_id, rig_camera);
    }

    void CameraRig::AddSnapshot(const std::vector<image_t> &image_ids) {
        CHECK(!image_ids.empty());
        CHECK_LE(image_ids.size(), NumCameras());
        CHECK(!VectorContainsDuplicateValues(image_ids));
        snapshots_.push_back(image_ids);
    }

    void CameraRig::AddSuffix(std::string &suffix) { snapshot_suffixs_.push_back(suffix); }
    void CameraRig::AddQvec(Eigen::Vector4d &qvec) { qvecs_.push_back(qvec); }
    void CameraRig::AddTvec(Eigen::Vector3d &tvec) { tvecs_.push_back(tvec); }

    void CameraRig::Check(const Reconstruction &reconstruction) const {
        CHECK(HasCamera(ref_camera_id_));

        for (const auto &rig_camera: rig_cameras_) {
            CHECK(reconstruction.ExistsCamera(rig_camera.first));
        }

        std::unordered_set<image_t> all_image_ids;
        for (const auto &snapshot: snapshots_) {
            CHECK(!snapshot.empty());
            CHECK_LE(snapshot.size(), NumCameras());
            bool has_ref_camera = false;
            for (const auto image_id: snapshot) {
                CHECK(reconstruction.ExistsImage(image_id));
                CHECK_EQ(all_image_ids.count(image_id), 0);
                all_image_ids.insert(image_id);
                const auto &image = reconstruction.Image(image_id);
                CHECK(HasCamera(image.CameraId()));
                if (image.CameraId() == ref_camera_id_) {
                    has_ref_camera = true;
                }
            }
            CHECK(has_ref_camera);
        }
    }

    Eigen::Vector4d &CameraRig::RelativeQvec(const camera_t camera_id) {
        return rig_cameras_.at(camera_id).rel_qvec;
    }

    const Eigen::Vector4d &CameraRig::RelativeQvec(const camera_t camera_id) const {
        return rig_cameras_.at(camera_id).rel_qvec;
    }

    Eigen::Vector3d &CameraRig::RelativeTvec(const camera_t camera_id) {
        return rig_cameras_.at(camera_id).rel_tvec;
    }

    const Eigen::Vector3d &CameraRig::RelativeTvec(const camera_t camera_id) const {
        return rig_cameras_.at(camera_id).rel_tvec;
    }

    double CameraRig::ComputeScale(const Reconstruction &reconstruction) const {
        CHECK_GT(NumSnapshots(), 0);
        CHECK_GT(NumCameras(), 0);
        double scaling_factor = 0;
        size_t num_dists      = 0;
        std::vector<Eigen::Vector3d> rel_proj_centers(NumCameras());
        std::vector<Eigen::Vector3d> abs_proj_centers(NumCameras());
        for (const auto &snapshot: snapshots_) {
            // Compute the projection relative and absolute projection centers.
            for (size_t i = 0; i < NumCameras(); ++i) {
                const auto &image   = reconstruction.Image(snapshot[i]);
                rel_proj_centers[i] = ProjectionCenterFromPose(
                        RelativeQvec(image.CameraId()), RelativeTvec(image.CameraId()));
                abs_proj_centers[i] = image.ProjectionCenter();
            }

            // Accumulate the scaling factor for all pairs of camera distances.
            for (size_t i = 0; i < NumCameras(); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    const double rel_dist =
                            (rel_proj_centers[i] - rel_proj_centers[j]).norm();
                    const double abs_dist =
                            (abs_proj_centers[i] - abs_proj_centers[j]).norm();
                    const double kMinDist = 1e-6;
                    if (rel_dist > kMinDist && abs_dist > kMinDist) {
                        scaling_factor += rel_dist / abs_dist;
                        num_dists += 1;
                    }
                }
            }
        }

        if (num_dists == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return scaling_factor / num_dists;
    }

    //计算每一个相机相对于参考相机的位姿
    bool CameraRig::ComputeRelativePoses(const Reconstruction &reconstruction) {
        CHECK_GT(NumSnapshots(), 0);               // 检查快照数量是否大于0
        CHECK_NE(ref_camera_id_, kInvalidCameraId);// 检查参考相机ID是否有效

        for (auto &rig_camera: rig_cameras_) {
            rig_camera.second.rel_tvec = Eigen::Vector3d::Zero();// 将相机的相对平移向量初始化为零向量
        }

        EIGEN_STL_UMAP(camera_t, std::vector<Eigen::Vector4d>)
        rel_qvecs;// 存储相对四元数的映射

        for (const auto &snapshot: snapshots_) {
            // Find the image of the reference camera in the current snapshot.
            const Image *ref_image = nullptr;
            if (!reconstruction.IsImageRegistered(snapshot[0])) {
                continue;// 如果快照中的第一个图像未注册，则跳过该快照
            }
            for (const auto image_id: snapshot) {

                const auto &image = reconstruction.Image(image_id);
                if (image.CameraId() == ref_camera_id_) {
                    ref_image = &image;
                }
            }

            CHECK_NOTNULL(ref_image);// 检查是否找到了参考图像

            // Compute the relative poses from all cameras in the current snapshot to the reference camera.
            for (const auto image_id: snapshot) {
                const auto &image = reconstruction.Image(image_id);
                if (image.CameraId() != ref_camera_id_) {
                    Eigen::Vector4d rel_qvec;
                    Eigen::Vector3d rel_tvec;
                    ComputeRelativePose(ref_image->Qvec(), ref_image->Tvec(), image.Qvec(),
                                        image.Tvec(), &rel_qvec, &rel_tvec);// 计算相对姿态

                    rel_qvecs[image.CameraId()].push_back(rel_qvec);// 将相对四元数添加到对应相机的列表中
                    RelativeTvec(image.CameraId()) += rel_tvec;     // 累加相对平移向量
                }
            }
        }

        RelativeQvec(ref_camera_id_) = ComposeIdentityQuaternion();// 设置参考相机的相对四元数为单位四元数
        RelativeTvec(ref_camera_id_) = Eigen::Vector3d(0, 0, 0);   // 设置参考相机的相对平移向量为零向量

        // Compute the average relative poses.
        for (auto &rig_camera: rig_cameras_) {
            if (rig_camera.first != ref_camera_id_) {
                if (rel_qvecs.count(rig_camera.first) == 0) {
                    std::cout << "Need at least one snapshot with an image of camera "
                              << rig_camera.first << " and the reference camera "
                              << ref_camera_id_
                              << " to compute its relative pose in the camera rig"
                              << std::endl;
                    return false;// 如果没有足够的快照来计算相机的相对姿态，则返回false
                }

                const std::vector<Eigen::Vector4d> &camera_rel_qvecs =
                        rel_qvecs.at(rig_camera.first);                                  // 获取相机的相对四元数列表
                const std::vector<double> rel_qvec_weights(camera_rel_qvecs.size(), 1.0);// 初始化权重列表为1.0
                rig_camera.second.rel_qvec =
                        AverageQuaternions(camera_rel_qvecs, rel_qvec_weights);// 计算相机的平均相对四元数
                rig_camera.second.rel_tvec /= camera_rel_qvecs.size();         // 计算相机的平均相对平移向量
                //std::cout << "\ncamera_rel_qvecs.size() = " << camera_rel_qvecs.size() << "\n";
            }
        }
        return true;// 返回计算相对姿态是否成功
    }

    //通过计算该snapshot时已注册的每一个相机得到的cam rig的位姿
    //进行平均得到相机组的绝对位姿
    void CameraRig::ComputeAbsolutePose(const size_t snapshot_idx,
                                        const Reconstruction &reconstruction,
                                        Eigen::Vector4d *abs_qvec,
                                        Eigen::Vector3d *abs_tvec) const {
        const auto &snapshot = snapshots_.at(snapshot_idx);

        std::vector<Eigen::Vector4d> abs_qvecs;
        *abs_tvec = Eigen::Vector3d::Zero();

        for (const auto image_id: snapshot) {
            const auto &image = reconstruction.Image(image_id);
            auto it           = std::find(reconstruction.RegImageIds().begin(), reconstruction.RegImageIds().end(), image_id);
            //如果该图像没有被注册，那么就跳过
            if (it == reconstruction.RegImageIds().end()) {
                continue;
            }
            Eigen::Vector4d inv_rel_qvec;
            Eigen::Vector3d inv_rel_tvec;
            InvertPose(RelativeQvec(image.CameraId()), RelativeTvec(image.CameraId()),
                       &inv_rel_qvec, &inv_rel_tvec);//得到该图片在世界坐标系下，相对于参考相机的位姿
            //相当于(world-camera)*(camera-relative)=world-relative，得到世界相对于参考相机的姿态
            const Eigen::Vector4d qvec =
                    ConcatenateQuaternions(image.Qvec(), inv_rel_qvec);
            //得到世界相对于参考相机的位置
            const Eigen::Vector3d tvec = QuaternionRotatePoint(
                    inv_rel_qvec, image.Tvec() - RelativeTvec(image.CameraId()));
            //记录一个快照中每一个相机计算得到的，参考相机的绝对位姿（世界相对参考相机）
            abs_qvecs.push_back(qvec);
            *abs_tvec += tvec;
        }
        //通过平均，得到该快照下，参考相机的平均位姿
        const std::vector<double> abs_qvec_weights(snapshot.size(), 1);
        *abs_qvec = AverageQuaternions(abs_qvecs, abs_qvec_weights);
        *abs_tvec /= snapshot.size();
    }

}// namespace colmap
