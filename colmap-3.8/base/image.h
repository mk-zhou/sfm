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

#ifndef COLMAP_SRC_BASE_IMAGE_H_
#define COLMAP_SRC_BASE_IMAGE_H_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/point2d.h"
#include "base/visibility_pyramid.h"
#include "util/alignment.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/types.h"

namespace colmap {

    // Class that holds information about an image. An image is the product of one
    // camera shot at a certain location (parameterized as the pose). An image may
    // share a camera with multiple other images, if its intrinsics are the same.
    class Image {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Image();

        // Setup / tear down the image and necessary internal data structures before
        // and after being used in reconstruction.
        void SetUp(const Camera &camera);
        void TearDown();

        // Access the unique identifier of the image.
        inline image_t ImageId() const;
        inline void SetImageId(const image_t image_id);

        inline my_snapshot_t SnapshotId() const;
        inline void SetSnapshotId(const my_snapshot_t image_id);

        // Access the name of the image.
        inline const std::string &Name() const;
        inline std::string &Name();
        inline void SetName(const std::string &name);

        // Access the unique identifier of the camera. Note that multiple images
        // might share the same camera.
        inline camera_t CameraId() const;
        inline void SetCameraId(const camera_t camera_id);
        // Check whether identifier of camera has been set.
        inline bool HasCamera() const;

        // Check if image is registered.
        inline bool IsRegistered() const;
        inline void SetRegistered(const bool registered);

        // Get the number of image points.
        inline point2D_t NumPoints2D() const;

        // Get the number of triangulations, i.e. the number of points that
        // are part of a 3D point track.
        inline point2D_t NumPoints3D() const;

        // Get the number of observations, i.e. the number of image points that
        // have at least one correspondence to another image.
        inline point2D_t NumObservations() const;
        inline void SetNumObservations(const point2D_t num_observations);

        // Get the number of correspondences for all image points.
        inline point2D_t NumCorrespondences() const;
        inline void SetNumCorrespondences(const point2D_t num_observations);

        // Get the number of observations that see a triangulated point, i.e. the
        // number of image points that have at least one correspondence to a
        // triangulated point in another image.
        inline point2D_t NumVisiblePoints3D() const;

        // Get the score of triangulated observations. In contrast to
        // `NumVisiblePoints3D`, this score also captures the distribution
        // of triangulated observations in the image. This is useful to select
        // the next best image in incremental reconstruction, because a more
        // uniform distribution of observations results in more robust registration.
        inline size_t Point3DVisibilityScore() const;

        // Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
        // pose which is defined as the transformation from world to image space.
        inline const Eigen::Vector4d &Qvec() const;
        inline Eigen::Vector4d &Qvec();
        inline double Qvec(const size_t idx) const;
        inline double &Qvec(const size_t idx);
        inline void SetQvec(const Eigen::Vector4d &qvec);

        // Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
        // pose which is defined as the transformation from world to image space.
        inline const Eigen::Vector4d &Ex_Rotation() const;
        inline Eigen::Vector4d &Ex_Rotation();
        inline double Ex_Rotation(const size_t idx) const;
        inline double &Ex_Rotation(const size_t idx);
        inline void SetEx_Rotation(const Eigen::Vector4d &qvec);


        inline const Eigen::Vector4d &Car_Rotation() const;
        inline Eigen::Vector4d &Car_Rotation();
        inline double Car_Rotation(const size_t idx) const;
        inline double &Car_Rotation(const size_t idx);
        inline void SetCar_Rotation(const Eigen::Vector4d &qvec);


        // Quaternion prior, e.g. given by EXIF gyroscope tag.
        inline const Eigen::Vector4d &QvecPrior() const;
        inline Eigen::Vector4d &QvecPrior();
        inline double QvecPrior(const size_t idx) const;
        inline double &QvecPrior(const size_t idx);
        inline bool HasQvecPrior() const;
        inline void SetQvecPrior(const Eigen::Vector4d &qvec);

        // Access translation vector as (tx, ty, tz) specifying the translation of the
        // pose which is defined as the transformation from world to image space.
        inline const Eigen::Vector3d &Tvec() const;
        inline Eigen::Vector3d &Tvec();
        inline double Tvec(const size_t idx) const;
        inline double &Tvec(const size_t idx);
        inline void SetTvec(const Eigen::Vector3d &tvec);

        inline const Eigen::Vector3d &Ex_Translation() const;
        inline Eigen::Vector3d &Ex_Translation();
        inline double Ex_Translation(const size_t idx) const;
        inline double &Ex_Translation(const size_t idx);
        inline void SetEx_Translation(const Eigen::Vector3d &tvec);

        inline const Eigen::Vector3d &Car_Translation() const;
        inline Eigen::Vector3d &Car_Translation();
        inline double Car_Translation(const size_t idx) const;
        inline double &Car_Translation(const size_t idx);
        inline void SetCar_Translation(const Eigen::Vector3d &tvec);

        //获取世界坐标系下的旋转矩阵与平移向量,四元数(qw, qx, qy, qz),平移向量(x,y,z)
        inline Eigen::Vector4d Qvec_world() const;
        inline Eigen::Matrix3d Rotation_world() const;
        inline Eigen::Vector3d Tvec_world() const;


        // Translation prior, e.g. given by EXIF GPS tag.
        inline const Eigen::Vector3d &TvecPrior() const;
        inline Eigen::Vector3d &TvecPrior();
        inline double TvecPrior(const size_t idx) const;
        inline double &TvecPrior(const size_t idx);
        inline bool HasTvecPrior() const;
        inline void SetTvecPrior(const Eigen::Vector3d &tvec);

        // Access the coordinates of image points.
        inline const class Point2D &Point2D(const point2D_t point2D_idx) const;
        inline class Point2D &Point2D(const point2D_t point2D_idx);
        inline const std::vector<class Point2D> &Points2D() const;
        void SetPoints2D(const std::vector<Eigen::Vector2d> &points);
        void SetPoints2D(const std::vector<class Point2D> &points);

        // Set the point as triangulated, i.e. it is part of a 3D point track.
        void SetPoint3DForPoint2D(const point2D_t point2D_idx,
                                  const point3D_t point3D_id);

        // Set the point as not triangulated, i.e. it is not part of a 3D point track.
        void ResetPoint3DForPoint2D(const point2D_t point2D_idx);

        // Check whether an image point has a correspondence to an image point in
        // another image that has a 3D point.
        inline bool IsPoint3DVisible(const point2D_t point2D_idx) const;

        // Check whether one of the image points is part of the 3D point track.
        bool HasPoint3D(const point3D_t point3D_id) const;

        // Indicate that another image has a point that is triangulated and has
        // a correspondence to this image point. Note that this must only be called
        // after calling `SetUp`.
        void IncrementCorrespondenceHasPoint3D(const point2D_t point2D_idx);

        // Indicate that another image has a point that is not triangulated any more
        // and has a correspondence to this image point. This assumes that
        // `IncrementCorrespondenceHasPoint3D` was called for the same image point
        // and correspondence before. Note that this must only be called
        // after calling `SetUp`.
        void DecrementCorrespondenceHasPoint3D(const point2D_t point2D_idx);

        // Normalize the quaternion vector.
        void NormalizeQvec();

        // Compose the projection matrix from world to image space.
        Eigen::Matrix3x4d ProjectionMatrix() const;

        // Compose the inverse projection matrix from image to world space
        Eigen::Matrix3x4d InverseProjectionMatrix() const;

        // Compose rotation matrix from quaternion vector.
        Eigen::Matrix3d RotationMatrix() const;

        // Extract the projection center in world space.
        Eigen::Vector3d ProjectionCenter() const;

        // Extract the viewing direction of the image.
        Eigen::Vector3d ViewingDirection() const;

        // The number of levels in the 3D point multi-resolution visibility pyramid.
        static const int kNumPoint3DVisibilityPyramidLevels;

    private:
        //快照
        my_snapshot_t snapshot_id_;

        //外参
        Eigen::Vector4d ex_rotation_;
        Eigen::Vector3d ex_translation_;

        //车身位姿
        Eigen::Vector4d car_rotation_;
        Eigen::Vector3d car_translation_;

        // Identifier of the image, if not specified `kInvalidImageId`.
        image_t image_id_;

        // The name of the image, i.e. the relative path.
        std::string name_;

        // The identifier of the associated camera. Note that multiple images might
        // share the same camera. If not specified `kInvalidCameraId`.
        camera_t camera_id_;

        // Whether the image is successfully registered in the reconstruction.
        bool registered_;

        // The number of 3D points the image observes, i.e. the sum of its `points2D`
        // where `point3D_id != kInvalidPoint3DId`.
        point2D_t num_points3D_;

        // The number of image points that have at least one correspondence to
        // another image.
        point2D_t num_observations_;

        // The sum of correspondences per image point.
        point2D_t num_correspondences_;

        // The number of 2D points, which have at least one corresponding 2D point in
        // another image that is part of a 3D point track, i.e. the sum of `points2D`
        // where `num_tris > 0`.
        point2D_t num_visible_points3D_;

        // The pose of the image, defined as the transformation from world to image.
        Eigen::Vector4d qvec_;
        Eigen::Vector3d tvec_;

        // The pose prior of the image, e.g. extracted from EXIF tags.
        Eigen::Vector4d qvec_prior_;
        Eigen::Vector3d tvec_prior_;

        //snapshot_id camera_id 从image类指向camera类和snapshot类

        // All image points, including points that are not part of a 3D point track.
        std::vector<class Point2D> points2D_;

        // Per image point, the number of correspondences that have a 3D point.
        std::vector<point2D_t> num_correspondences_have_point3D_;

        // Data structure to compute the distribution of triangulated correspondences
        // in the image. Note that this structure is only usable after `SetUp`.
        VisibilityPyramid point3D_visibility_pyramid_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Implementation
    ////////////////////////////////////////////////////////////////////////////////

    image_t Image::ImageId() const { return image_id_; }

    void Image::SetImageId(const image_t image_id) { image_id_ = image_id; }

    my_snapshot_t Image::SnapshotId() const { return snapshot_id_; }

    void Image::SetSnapshotId(const my_snapshot_t snapshot_id) { snapshot_id_ = snapshot_id; }

    const std::string &Image::Name() const { return name_; }

    std::string &Image::Name() { return name_; }

    void Image::SetName(const std::string &name) { name_ = name; }

    inline camera_t Image::CameraId() const { return camera_id_; }

    inline void Image::SetCameraId(const camera_t camera_id) {
        CHECK_NE(camera_id, kInvalidCameraId);
        camera_id_ = camera_id;
    }

    inline bool Image::HasCamera() const { return camera_id_ != kInvalidCameraId; }

    bool Image::IsRegistered() const { return registered_; }

    void Image::SetRegistered(const bool registered) { registered_ = registered; }

    point2D_t Image::NumPoints2D() const {
        return static_cast<point2D_t>(points2D_.size());
    }

    point2D_t Image::NumPoints3D() const { return num_points3D_; }

    point2D_t Image::NumObservations() const { return num_observations_; }

    void Image::SetNumObservations(const point2D_t num_observations) {
        num_observations_ = num_observations;
    }

    point2D_t Image::NumCorrespondences() const { return num_correspondences_; }

    void Image::SetNumCorrespondences(const point2D_t num_correspondences) {
        num_correspondences_ = num_correspondences;
    }

    point2D_t Image::NumVisiblePoints3D() const { return num_visible_points3D_; }

    size_t Image::Point3DVisibilityScore() const {

        return point3D_visibility_pyramid_.Score();
    }


    const Eigen::Vector4d &Image::Ex_Rotation() const { return ex_rotation_; }

    Eigen::Vector4d &Image::Ex_Rotation() { return ex_rotation_; }

    inline double Image::Ex_Rotation(const size_t idx) const { return ex_rotation_(idx); }

    inline double &Image::Ex_Rotation(const size_t idx) { return ex_rotation_(idx); }

    void Image::SetEx_Rotation(const Eigen::Vector4d &ex_rotation) { ex_rotation_ = ex_rotation; }


    const Eigen::Vector4d &Image::Car_Rotation() const { return car_rotation_; }

    Eigen::Vector4d &Image::Car_Rotation() { return car_rotation_; }

    inline double Image::Car_Rotation(const size_t idx) const { return car_rotation_(idx); }

    inline double &Image::Car_Rotation(const size_t idx) { return car_rotation_(idx); }

    void Image::SetCar_Rotation(const Eigen::Vector4d &car_rotation) { car_rotation_ = car_rotation; }


    const Eigen::Vector4d &Image::Qvec() const { return qvec_; }

    Eigen::Vector4d &Image::Qvec() { return qvec_; }

    inline double Image::Qvec(const size_t idx) const { return qvec_(idx); }

    inline double &Image::Qvec(const size_t idx) { return qvec_(idx); }

    void Image::SetQvec(const Eigen::Vector4d &qvec) { qvec_ = qvec; }


    const Eigen::Vector4d &Image::QvecPrior() const { return qvec_prior_; }

    Eigen::Vector4d &Image::QvecPrior() { return qvec_prior_; }

    inline double Image::QvecPrior(const size_t idx) const {
        return qvec_prior_(idx);
    }

    inline double &Image::QvecPrior(const size_t idx) { return qvec_prior_(idx); }

    inline bool Image::HasQvecPrior() const { return !IsNaN(qvec_prior_.sum()); }

    void Image::SetQvecPrior(const Eigen::Vector4d &qvec) { qvec_prior_ = qvec; }


    const Eigen::Vector3d &Image::Ex_Translation() const { return ex_translation_; }

    Eigen::Vector3d &Image::Ex_Translation() { return ex_translation_; }

    inline double Image::Ex_Translation(const size_t idx) const { return ex_translation_(idx); }

    inline double &Image::Ex_Translation(const size_t idx) { return ex_translation_(idx); }

    void Image::SetEx_Translation(const Eigen::Vector3d &ex_translation) { ex_translation_ = ex_translation; }


    const Eigen::Vector3d &Image::Car_Translation() const { return car_translation_; }

    Eigen::Vector3d &Image::Car_Translation() { return car_translation_; }

    inline double Image::Car_Translation(const size_t idx) const { return car_translation_(idx); }

    inline double &Image::Car_Translation(const size_t idx) { return car_translation_(idx); }

    void Image::SetCar_Translation(const Eigen::Vector3d &car_translation) { car_translation_ = car_translation; }


    const Eigen::Vector3d &Image::Tvec() const { return tvec_; }

    Eigen::Vector3d &Image::Tvec() { return tvec_; }

    inline double Image::Tvec(const size_t idx) const { return tvec_(idx); }

    inline double &Image::Tvec(const size_t idx) { return tvec_(idx); }

    void Image::SetTvec(const Eigen::Vector3d &tvec) { tvec_ = tvec; }


    Eigen::Vector4d Image::Qvec_world() const {
        Eigen::Quaterniond q(qvec_(0), qvec_(1), qvec_(2), qvec_(3));
        // 计算旋转矩阵
        Eigen::Matrix3d R = q.toRotationMatrix();

        // 计算逆矩阵
        Eigen::Matrix3d Rotation = R.inverse();

        // 计算对应的四元数
        Eigen::Quaterniond q_inv(Rotation);

        Eigen::Vector4d Qvec_world = q_inv.coeffs();

        return Qvec_world;
    }


    Eigen::Matrix3d Image::Rotation_world() const {
        Eigen::Quaterniond q(qvec_(0), qvec_(1), qvec_(2), qvec_(3));
        // 计算旋转矩阵
        Eigen::Matrix3d R = q.toRotationMatrix();

        // 计算逆矩阵
        Eigen::Matrix3d Rotation = R.inverse();

        return Rotation;
    }


    Eigen::Vector3d Image::Tvec_world() const {
        return (-this->Rotation_world() * tvec_);
    }

    const Eigen::Vector3d &Image::TvecPrior() const { return tvec_prior_; }

    Eigen::Vector3d &Image::TvecPrior() { return tvec_prior_; }

    inline double Image::TvecPrior(const size_t idx) const {
        return tvec_prior_(idx);
    }

    inline double &Image::TvecPrior(const size_t idx) { return tvec_prior_(idx); }

    inline bool Image::HasTvecPrior() const { return !IsNaN(tvec_prior_.sum()); }

    void Image::SetTvecPrior(const Eigen::Vector3d &tvec) { tvec_prior_ = tvec; }

    const class Point2D &Image::Point2D(const point2D_t point2D_idx) const {
        return points2D_.at(point2D_idx);
    }

    class Point2D &Image::Point2D(const point2D_t point2D_idx) {
        return points2D_.at(point2D_idx);
    }

    const std::vector<class Point2D> &Image::Points2D() const { return points2D_; }

    bool Image::IsPoint3DVisible(const point2D_t point2D_idx) const {
        return num_correspondences_have_point3D_.at(point2D_idx) > 0;
    }

}// namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::Image)

#endif// COLMAP_SRC_BASE_IMAGE_H_
