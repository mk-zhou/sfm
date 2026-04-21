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

#ifndef COLMAP_SRC_BASE_CAMERA_H_
#define COLMAP_SRC_BASE_CAMERA_H_

#include <vector>

#include "util/math.h"
#include "util/types.h"

namespace colmap {

    // Camera class that holds the intrinsic parameters. Cameras may be shared
    // between multiple images, e.g., if the same "physical" camera took multiple
    // pictures with the exact same lens and intrinsics (focal length, etc.).
    // This class has a specific distortion model defined by a camera model class.
    class Camera {
    public:
        Camera();

        // Access the unique identifier of the camera.
        inline camera_t CameraId() const;
        inline void SetCameraId(const camera_t camera_id);

        // Access the camera model.
        inline int ModelId() const;
        std::string ModelName() const;
        void SetModelId(const int model_id);
        void SetModelIdFromName(const std::string &model_name);

        // Access dimensions of the camera sensor.
        inline size_t Width() const;
        inline size_t Height() const;
        inline void SetWidth(const size_t width);
        inline void SetHeight(const size_t height);

        // Access focal length parameters.
        double MeanFocalLength() const;
        double FocalLength() const;
        double FocalLengthX() const;
        double FocalLengthY() const;
        void SetFocalLength(const double focal_length);
        void SetFocalLengthX(const double focal_length_x);
        void SetFocalLengthY(const double focal_length_y);

        // Check if camera has prior focal length.
        inline bool HasPriorFocalLength() const;
        inline void SetPriorFocalLength(const bool prior);

        // Access principal point parameters. Only works if there are two
        // principal point parameters.
        double PrincipalPointX() const;
        double PrincipalPointY() const;
        void SetPrincipalPointX(const double ppx);
        void SetPrincipalPointY(const double ppy);

        // Get the indices of the parameter groups in the parameter vector.
        const std::vector<size_t> &FocalLengthIdxs() const;
        const std::vector<size_t> &PrincipalPointIdxs() const;
        const std::vector<size_t> &ExtraParamsIdxs() const;

        // Get intrinsic calibration matrix composed from focal length and principal
        // point parameters, excluding distortion parameters.
        Eigen::Matrix3d CalibrationMatrix() const;

        // Get human-readable information about the parameter vector ordering.
        std::string ParamsInfo() const;

        // Access the raw parameter vector.
        inline size_t NumParams() const;
        inline const std::vector<double> &Params() const;
        inline std::vector<double> &Params();
        inline double Params(const size_t idx) const;
        inline double &Params(const size_t idx);
        inline const double *ParamsData() const;
        inline double *ParamsData();
        inline void SetParams(const std::vector<double> &params);

        // Concatenate parameters as comma-separated list.
        std::string ParamsToString() const;

        // Set camera parameters from comma-separated list.
        bool SetParamsFromString(const std::string &string);

        // Check whether parameters are valid, i.e. the parameter vector has
        // the correct dimensions that match the specified camera model.
        bool VerifyParams() const;

        // Check whether camera is already undistorted
        bool IsUndistorted() const;

        // Check whether camera has bogus parameters.
        bool HasBogusParams(const double min_focal_length_ratio,
                            const double max_focal_length_ratio,
                            const double max_extra_param) const;

        // Initialize parameters for given camera model and focal length, and set
        // the principal point to be the image center.
        void InitializeWithId(const int model_id, const double focal_length,
                              const size_t width, const size_t height);
        void InitializeWithName(const std::string &model_name,
                                const double focal_length, const size_t width,
                                const size_t height);

        // Project point in image plane to world / infinity.
        Eigen::Vector2d ImageToWorld(const Eigen::Vector2d &image_point) const;

        // Convert pixel threshold in image plane to world space.
        double ImageToWorldThreshold(const double threshold) const;

        // Project point from world / infinity to image plane.
        Eigen::Vector2d WorldToImage(const Eigen::Vector2d &world_point) const;

        // Rescale camera dimensions and accordingly the focal length and
        // and the principal point.
        void Rescale(const double scale);
        void Rescale(const size_t width, const size_t height);

        inline const Eigen::Vector4d &ExQvec() const;
        inline Eigen::Vector4d &ExQvec();
        inline double ExQvec(const size_t idx) const;
        inline double &ExQvec(const size_t idx);
        inline void SetExQvec(const Eigen::Vector4d &ex_qvec);

        // Quaternion prior, e.g. given by EXIF gyroscope tag.
        inline const Eigen::Vector4d &ExQvecPrior() const;
        inline Eigen::Vector4d &ExQvecPrior();
        inline double ExQvecPrior(const size_t idx) const;
        inline double &ExQvecPrior(const size_t idx);
        inline bool HasExQvecPrior() const;
        inline void SetExQvecPrior(const Eigen::Vector4d &ex_qvec);

        inline const Eigen::Vector3d &ExTvec() const;
        inline Eigen::Vector3d &ExTvec();
        inline double ExTvec(const size_t idx) const;
        inline double &ExTvec(const size_t idx);
        inline void SetExTvec(const Eigen::Vector3d &ex_tvec);

        // Quaternion prior, e.g. given by EXIF gyroscope tag.
        inline const Eigen::Vector3d &ExTvecPrior() const;
        inline Eigen::Vector3d &ExTvecPrior();
        inline double ExTvecPrior(const size_t idx) const;
        inline double &ExTvecPrior(const size_t idx);
        inline bool HasExTvecPrior() const;
        inline void SetExTvecPrior(const Eigen::Vector3d &ex_tvec);

    private:
        // The unique identifier of the camera. If the identifier is not specified
        // it is set to `kInvalidCameraId`.
        camera_t camera_id_;

        // The identifier of the camera model. If the camera model is not specified
        // the identifier is `kInvalidCameraModelId`.
        int model_id_;

        // The dimensions of the image, 0 if not initialized.
        size_t width_;
        size_t height_;

        // The focal length, principal point, and extra parameters. If the camera
        // model is not specified, this vector is empty.
        std::vector<double> params_;

        // Whether there is a safe prior for the focal length,
        // e.g. manually provided or extracted from EXIF
        bool prior_focal_length_;

        // 该相机相对车身的旋转与位移
        Eigen::Vector4d ex_qvec_;
        Eigen::Vector3d ex_tvec_;

        // 从database里面得到的先验
        Eigen::Vector4d ex_qvec_prior_;
        Eigen::Vector3d ex_tvec_prior_;
    };

    class MySnapshot {
    public:
        //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MySnapshot();

        // Setup / tear down the image and necessary internal data structures before
        // and after being used in reconstruction.
        //void SetUp(const Camera& camera);
        //void TearDown();

        // Access the unique identifier of the image.
        inline my_snapshot_t MySnapshotId() const;
        inline void SetMySnapshotId(const my_snapshot_t my_snapshot_id);

        // Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
        // pose which is defined as the transformation from world to image space.
        inline const Eigen::Vector4d &Qvec() const;
        inline Eigen::Vector4d &Qvec();
        inline double Qvec(const size_t idx) const;
        inline double &Qvec(const size_t idx);
        inline void SetQvec(const Eigen::Vector4d &qvec);

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

        // Translation prior, e.g. given by EXIF GPS tag.
        inline const Eigen::Vector3d &TvecPrior() const;
        inline Eigen::Vector3d &TvecPrior();
        inline double TvecPrior(const size_t idx) const;
        inline double &TvecPrior(const size_t idx);
        inline bool HasTvecPrior() const;
        inline void SetTvecPrior(const Eigen::Vector3d &tvec);

        // Normalize the quaternion vector.
        void NormalizeQvec();

    private:
        my_snapshot_t my_snapshot_id_;
        // The pose of the snapshot, defined as the transformation from world to car.
        Eigen::Vector4d qvec_;
        Eigen::Vector3d tvec_;

        // The pose prior of the snapshot
        Eigen::Vector4d qvec_prior_;
        Eigen::Vector3d tvec_prior_;
    };
    ////////////////////////////////////////////////////////////////////////////////
    // Implementation
    ////////////////////////////////////////////////////////////////////////////////

    camera_t Camera::CameraId() const { return camera_id_; }

    void Camera::SetCameraId(const camera_t camera_id) { camera_id_ = camera_id; }

    int Camera::ModelId() const { return model_id_; }

    size_t Camera::Width() const { return width_; }

    size_t Camera::Height() const { return height_; }

    void Camera::SetWidth(const size_t width) { width_ = width; }

    void Camera::SetHeight(const size_t height) { height_ = height; }

    bool Camera::HasPriorFocalLength() const { return prior_focal_length_; }

    void Camera::SetPriorFocalLength(const bool prior) {
        prior_focal_length_ = prior;
    }

    size_t Camera::NumParams() const { return params_.size(); }

    const std::vector<double> &Camera::Params() const { return params_; }

    std::vector<double> &Camera::Params() { return params_; }

    double Camera::Params(const size_t idx) const { return params_[idx]; }

    double &Camera::Params(const size_t idx) { return params_[idx]; }

    const double *Camera::ParamsData() const { return params_.data(); }

    double *Camera::ParamsData() { return params_.data(); }

    void Camera::SetParams(const std::vector<double> &params) { params_ = params; }


    const Eigen::Vector4d &Camera::ExQvec() const { return ex_qvec_; }

    Eigen::Vector4d &Camera::ExQvec() { return ex_qvec_; }

    inline double Camera::ExQvec(const size_t idx) const { return ex_qvec_(idx); }

    inline double &Camera::ExQvec(const size_t idx) { return ex_qvec_(idx); }

    void Camera::SetExQvec(const Eigen::Vector4d &ex_qvec) { ex_qvec_ = ex_qvec; }


    const Eigen::Vector4d &Camera::ExQvecPrior() const { return ex_qvec_prior_; }

    Eigen::Vector4d &Camera::ExQvecPrior() { return ex_qvec_prior_; }

    inline double Camera::ExQvecPrior(const size_t idx) const {
        return ex_qvec_prior_(idx);
    }

    inline double &Camera::ExQvecPrior(const size_t idx) { return ex_qvec_prior_(idx); }

    inline bool Camera::HasExQvecPrior() const { return !IsNaN(ex_qvec_prior_.sum()); }

    void Camera::SetExQvecPrior(const Eigen::Vector4d &ex_qvec) { ex_qvec_prior_ = ex_qvec; }


    const Eigen::Vector3d &Camera::ExTvec() const { return ex_tvec_; }

    Eigen::Vector3d &Camera::ExTvec() { return ex_tvec_; }

    inline double Camera::ExTvec(const size_t idx) const { return ex_tvec_(idx); }

    inline double &Camera::ExTvec(const size_t idx) { return ex_tvec_(idx); }

    void Camera::SetExTvec(const Eigen::Vector3d &ex_tvec) { ex_tvec_ = ex_tvec; }


    const Eigen::Vector3d &Camera::ExTvecPrior() const { return ex_tvec_prior_; }

    Eigen::Vector3d &Camera::ExTvecPrior() { return ex_tvec_prior_; }

    inline double Camera::ExTvecPrior(const size_t idx) const {
        return ex_tvec_prior_(idx);
    }

    inline double &Camera::ExTvecPrior(const size_t idx) { return ex_tvec_prior_(idx); }

    inline bool Camera::HasExTvecPrior() const { return !IsNaN(ex_tvec_prior_.sum()); }

    void Camera::SetExTvecPrior(const Eigen::Vector3d &ex_tvec) { ex_tvec_prior_ = ex_tvec; }

    my_snapshot_t MySnapshot::MySnapshotId() const { return my_snapshot_id_; }

    void MySnapshot::SetMySnapshotId(const my_snapshot_t my_snapshot_id) { my_snapshot_id_ = my_snapshot_id; }

    const Eigen::Vector4d &MySnapshot::Qvec() const { return qvec_; }

    Eigen::Vector4d &MySnapshot::Qvec() { return qvec_; }

    inline double MySnapshot::Qvec(const size_t idx) const { return qvec_(idx); }

    inline double &MySnapshot::Qvec(const size_t idx) { return qvec_(idx); }

    void MySnapshot::SetQvec(const Eigen::Vector4d &qvec) { qvec_ = qvec; }


    const Eigen::Vector4d &MySnapshot::QvecPrior() const { return qvec_prior_; }

    Eigen::Vector4d &MySnapshot::QvecPrior() { return qvec_prior_; }

    inline double MySnapshot::QvecPrior(const size_t idx) const {
        return qvec_prior_(idx);
    }

    inline double &MySnapshot::QvecPrior(const size_t idx) { return qvec_prior_(idx); }

    inline bool MySnapshot::HasQvecPrior() const { return !IsNaN(qvec_prior_.sum()); }

    void MySnapshot::SetQvecPrior(const Eigen::Vector4d &qvec) { qvec_prior_ = qvec; }


    const Eigen::Vector3d &MySnapshot::Tvec() const { return tvec_; }

    Eigen::Vector3d &MySnapshot::Tvec() { return tvec_; }

    inline double MySnapshot::Tvec(const size_t idx) const { return tvec_(idx); }

    inline double &MySnapshot::Tvec(const size_t idx) { return tvec_(idx); }

    void MySnapshot::SetTvec(const Eigen::Vector3d &tvec) { tvec_ = tvec; }


    const Eigen::Vector3d &MySnapshot::TvecPrior() const { return tvec_prior_; }

    Eigen::Vector3d &MySnapshot::TvecPrior() { return tvec_prior_; }

    inline double MySnapshot::TvecPrior(const size_t idx) const {
        return tvec_prior_(idx);
    }

    inline double &MySnapshot::TvecPrior(const size_t idx) { return tvec_prior_(idx); }

    inline bool MySnapshot::HasTvecPrior() const { return !IsNaN(tvec_prior_.sum()); }

    void MySnapshot::SetTvecPrior(const Eigen::Vector3d &tvec) { tvec_prior_ = tvec; }

}// namespace colmap

#endif// COLMAP_SRC_BASE_CAMERA_H_
