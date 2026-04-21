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

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

    template <typename Scalar>
    Eigen::Matrix<Scalar, 3, 1> safeEulerAngles(const Eigen::Quaternion<Scalar>& quat) {
        Eigen::Matrix<Scalar, 3, 1> euler_error;
        // 定义一个非常小的阈值，用于数值误差比较
        const double EPSILON = 1e-6;
        // 使用 ceres::abs 替代 std::abs
        if (ceres::abs(quat.w() - Scalar(1.0)) < Scalar(EPSILON) && ceres::abs(quat.x()) < Scalar(EPSILON) &&
            ceres::abs(quat.y()) < Scalar(EPSILON) && ceres::abs(quat.z()) < Scalar(EPSILON)) {
            // 使用正确的赋值方式
            euler_error << Scalar(1e-16), Scalar(1e-16), Scalar(1e-16);
        } else {
            // 正常转换为欧拉角
            euler_error = quat.toRotationMatrix().eulerAngles(2, 1, 0).template cast<Scalar>();
        }

        return euler_error;
    }

    template <typename Scalar>
    Eigen::Matrix<Scalar, 3, 1> safe_quat2axis(const Eigen::Quaternion<Scalar>& quat) {
        Eigen::Matrix<Scalar, 3, 1> angle_axis;
        // 定义一个非常小的阈值，用于数值误差比较
        const double EPSILON = 1e-6;
        // 使用 ceres::abs 替代 std::abs
        if (ceres::abs(ceres::abs(quat.w()) - Scalar(1.0)) < Scalar(EPSILON) && ceres::abs(quat.x()) < Scalar(EPSILON) &&
            ceres::abs(quat.y()) < Scalar(EPSILON) && ceres::abs(quat.z()) < Scalar(EPSILON)) {
            // 使用正确的赋值方式
            angle_axis << Scalar(1e-16), Scalar(1e-16), Scalar(1e-16);
        } else {
            std::cout<<"quat: "<<quat.w()<<" "<<quat.x()<<" "<<quat.y()<<" "<<quat.z()<<std::endl;
            ceres::QuaternionToAngleAxis(quat.coeffs().data(), angle_axis.data());
        }

        return angle_axis;
    }


    // Standard bundle adjustment cost function for variable
    // camera pose and calibration and point parameters.
    // 所有元素都进行优化
    template<typename CameraModel>
    class BundleAdjustmentCostFunction {
    public:
        explicit BundleAdjustmentCostFunction(const Eigen::Vector2d &point2D)
            : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D) {
            return (new ceres::AutoDiffCostFunction<
                    BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
                    CameraModel::kNumParams>(
                    new BundleAdjustmentCostFunction(point2D)));
        }

        template<typename T>
        bool operator()(const T *const qvec, const T *const tvec,
                        const T *const point3D, const T *const camera_params,
                        T *residuals) const {
            // Rotate and translate.
            // 将三维点变化到相机坐标系中的投影点projection
            // 用world_to_camera表示位姿的原因：便于计算投影点
            T projection[3];

            /////////////////////////////
            //T point3D_trans[3];
            //point3D_trans[0] = point3D[0] - tvec[0];
            //point3D_trans[1] = point3D[1] - tvec[1];
            //point3D_trans[2] = point3D[2] - tvec[2];
            //ceres::UnitQuaternionRotatePoint(qvec, point3D_trans, projection);
            //上为添加内容
            ///////////////////////////////

            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            // 投影点归一化到图像平面
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            // 去畸变后的像素坐标
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            // 得到重投影误差
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
    };

    // Bundle adjustment cost function for variable
    // camera calibration and point parameters, and fixed camera pose.
    // 相机外参固定，其他用于优化
    template<typename CameraModel>
    class BundleAdjustmentConstantPoseCostFunction {
    public:
        BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d &qvec,
                                                 const Eigen::Vector3d &tvec,
                                                 const Eigen::Vector2d &point2D)
            : qw_(qvec(0)),
              qx_(qvec(1)),
              qy_(qvec(2)),
              qz_(qvec(3)),
              tx_(tvec(0)),
              ty_(tvec(1)),
              tz_(tvec(2)),
              observed_x_(point2D(0)),
              observed_y_(point2D(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector4d &qvec,
                                           const Eigen::Vector3d &tvec,
                                           const Eigen::Vector2d &point2D) {
            return (new ceres::AutoDiffCostFunction<
                    BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
                    CameraModel::kNumParams>(
                    new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
        }


        template<typename T>
        bool operator()(const T *const point3D, const T *const camera_params,
                        T *residuals) const {
            const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

            // Rotate and translate.
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += T(tx_);
            projection[1] += T(ty_);
            projection[2] += T(tz_);

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            return true;
        }

    private:
        //位姿保持不变
        const double qw_;
        const double qx_;
        const double qy_;
        const double qz_;
        const double tx_;
        const double ty_;
        const double tz_;
        const double observed_x_;
        const double observed_y_;
    };


    // Rig bundle adjustment cost function for variable camera pose and calibration
    // and point parameters. Different from the standard bundle adjustment function,
    // this cost function is suitable for camera rigs with consistent relative poses
    // of the cameras within the rig. The cost function first projects points into
    // the local system of the camera rig and then into the local system of the
    // camera within the rig.
    // 用于处理相机相对位姿固定的情况，首先将三维点投影到相机组内的局部坐标系
    // 然后从将机组的坐标系转换到相机坐标系
    template<typename CameraModel>
    class RigBundleAdjustmentCostFunction {
    public:
        explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d &point2D)
            : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D) {
            return (new ceres::AutoDiffCostFunction<
                    RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
                    CameraModel::kNumParams>(
                    new RigBundleAdjustmentCostFunction(point2D)));
        }

        template<typename T>
        bool operator()(const T *const rig_qvec, const T *const rig_tvec,
                        const T *const rel_qvec, const T *const rel_tvec,
                        const T *const point3D, const T *const camera_params,
                        T *residuals) const {
            // 得到世界相对于相机的q
            T qvec[4];
            ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

            // 得到世界相对于相机的t
            T tvec[3];
            ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
            tvec[0] += rel_tvec[0];
            tvec[1] += rel_tvec[1];
            tvec[2] += rel_tvec[2];

            // 变换到相机平面
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // 利用相机模型，转换到像素平面
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
    };


    // 用于处理Merges时的BA
    // 用先验位姿加上相对位姿来表示目前的位姿
    template<typename CameraModel>
    class MergeBundleAdjustmentCostFunction {
    public:
        explicit MergeBundleAdjustmentCostFunction(const Eigen::Vector2d &point2D)
            : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D) {
            return (new ceres::AutoDiffCostFunction<
                    MergeBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
                    CameraModel::kNumParams>(
                    new MergeBundleAdjustmentCostFunction(point2D)));
        }

        template<typename T>
        bool operator()(const T *const trans_qvec, const T *const trans_tvec,
                        const T *const prior_qvec, const T *const prior_tvec,
                        const T *const point3D, const T *const camera_params,
                        T *residuals) const {
            // 得到世界相对于相机的q
            T qvec[4];
            ceres::QuaternionProduct(prior_qvec, trans_qvec, qvec);

            // 得到世界相对于相机的t
            T tvec[3];
            ceres::UnitQuaternionRotatePoint(prior_qvec, trans_tvec, tvec);
            tvec[0] += prior_tvec[0];
            tvec[1] += prior_tvec[1];
            tvec[2] += prior_tvec[2];

            // 变换到相机平面
            T projection[3];
            ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // 利用相机模型，转换到像素平面
            CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                      &residuals[0], &residuals[1]);

            // Re-projection error.
            residuals[0] -= T(observed_x_);
            residuals[1] -= T(observed_y_);

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
    };


    // Cost function for refining two-view geometry based on the Sampson-Error.
    //
    // First pose is assumed to be located at the origin with 0 rotation. Second
    // pose is assumed to be on the unit sphere around the first pose, i.e. the
    // pose of the second camera is parameterized by a 3D rotation and a
    // 3D translation with unit norm. `tvec` is therefore over-parameterized as is
    // and should be down-projected using `SphereManifold`.
    class RelativePoseCostFunction {
    public:
        RelativePoseCostFunction(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2)
            : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &x1,
                                           const Eigen::Vector2d &x2) {
            return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
                    new RelativePoseCostFunction(x1, x2)));
        }

        template<typename T>
        bool operator()(const T *const qvec, const T *const tvec,
                        T *residuals) const {
            Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
            ceres::QuaternionToRotation(qvec, R.data());

            // Matrix representation of the cross product t x R.
            Eigen::Matrix<T, 3, 3> t_x;
            t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
                    T(0);

            // Essential matrix.
            const Eigen::Matrix<T, 3, 3> E = t_x * R;

            // Homogeneous image coordinates.
            const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
            const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

            // Squared sampson error.
            const Eigen::Matrix<T, 3, 1> Ex1  = E * x1_h;
            const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
            const T x2tEx1                    = x2_h.transpose() * Ex1;
            residuals[0]                      = x2tEx1 * x2tEx1 /
                           (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                            Etx2(1) * Etx2(1));

            return true;
        }

    private:
        const double x1_;
        const double y1_;
        const double x2_;
        const double y2_;
    };

    // 残差块用于优化位置和旋转误差
    struct PoseConstraintCostFunction {
        Eigen::Vector4d weight_;
        Eigen::Vector4d prior_qvec_;
        Eigen::Vector3d prior_tvec_;

        PoseConstraintCostFunction(const Eigen::Vector4d& qvec_prior, const Eigen::Vector3d& tvec_prior,
                                   const Eigen::Vector4d& weight)
            : prior_qvec_(qvec_prior), prior_tvec_(tvec_prior),
              weight_(weight) {}

        static ceres::CostFunction* Create(const Eigen::Vector4d& qvec_prior, const Eigen::Vector3d& tvec_prior,
                                           const Eigen::Vector4d& weight) {
            return (new ceres::AutoDiffCostFunction<PoseConstraintCostFunction, 4, 4, 3>(
                    new PoseConstraintCostFunction(qvec_prior, tvec_prior, weight)));
        }

        template <typename T>
        bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
            using Vec3T = Eigen::Matrix<T, 3, 1>;
            // 得到世界坐标系下的当前的相机位姿
            const Eigen::Quaternion<T> qcw_current(qvec[0], -qvec[1], -qvec[2], -qvec[3]);
            Vec3T t_current(tvec[0], tvec[1], tvec[2]);
            Vec3T tcw_current = qcw_current * -t_current;

            // 得到世界坐标系下的先验的相机位姿
            const Eigen::Quaternion<T> qcw_prior(
                    T(prior_qvec_(0)),
                    T(-prior_qvec_(1)),
                    T(-prior_qvec_(2)),
                    T(-prior_qvec_(3))
            );
            Vec3T t_prior(T(prior_tvec_(0)), T(prior_tvec_(1)), T(prior_tvec_(2)));
            Vec3T tcw_prior = qcw_prior * -t_prior;

            // 计算平移残差
            Vec3T t_error = tcw_prior - tcw_current;

            // 计算四元数的误差
            Eigen::Quaternion<T> q_diff = qcw_prior.conjugate() * qcw_current;

            // 将四元数转换为角轴形式
            Eigen::Matrix<T, 3, 1> angle_axis;
            angle_axis = safe_quat2axis(q_diff);
            // 将位置和旋转误差合并到残差中
            residuals[0] = weight_(0) * t_error(0);
            residuals[1] = weight_(1) * t_error(1);
            residuals[2] = weight_(2) * t_error(2);
            residuals[3] = weight_(3) * angle_axis.norm();

            return true;
        }
    };




    // Ceres CostFunctor used for SfM pose center to GPS pose center minimization
    // Ref: openMVG/sfm/sfm_data_BA_ceres.cpp
    struct PoseCenterConstraintCostFunction {
        Eigen::Vector3d weight_;
        Eigen::Vector3d pose_center_constraint_;

        PoseCenterConstraintCostFunction(
                const Eigen::Vector3d &center,
                const Eigen::Vector3d &weight) : weight_(weight), pose_center_constraint_(center) {
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &pose_center_constraint,
                                           const Eigen::Vector3d &weight) {
            return (new ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 4, 3>(
                    new PoseCenterConstraintCostFunction(pose_center_constraint, weight)));
        }

        template<typename T>
        bool
        operator()(
                const T *const qvec,// qcw
                const T *const tvec,// tcw
                T *residuals)
                const {
            using Vec3T = Eigen::Matrix<T, 3, 1>;

            Vec3T pose_center;// twc_sfm

            // Rotate the point according the camera rotation
            // Inverse rotation as conjugate quaternion.
            const Eigen::Quaternion<T> qcw(qvec[0], -qvec[1], -qvec[2], -qvec[3]);
            Vec3T tcw(tvec[0], tvec[1], tvec[2]);
            pose_center = qcw * -tcw;

            // Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R; //Rcw
            // ceres::QuaternionToRotation(qvec, R.data());
            // Eigen::Matrix<T, 3, 3, Eigen::RowMajor> Rwc;
            // Rwc = R.inverse();
            // T qwc[4];
            // ceres::RotationMatrixToQuaternion(Rwc.data(),qwc);
            // ceres::QuaternionRotatePoint(qwc,tvec,pose_center.data());
            // pose_center = pose_center * T(-1); //twc

            Eigen::Map<Vec3T> residuals_eigen(residuals);
            residuals_eigen = weight_.cast<T>().cwiseProduct(pose_center - pose_center_constraint_.cast<T>());

            return true;
        }
    };


    // sfm姿态和先验姿态构建损失
    // Ref: openMVG/sfm/sfm_data_BA_ceres.cpp
    struct PriorRotationConstraintCostFunction {
        Eigen::Vector3d weight_;
        Eigen::Vector4d prior_qvec_;

        PriorRotationConstraintCostFunction(
                const Eigen::Vector4d &prior_qvec,
                const Eigen::Vector3d &weight) : weight_(weight), prior_qvec_(prior_qvec) {
        }

        static ceres::CostFunction *Create(const Eigen::Vector4d &prior_qvec,
                                           const Eigen::Vector3d &weight) {
            return (new ceres::AutoDiffCostFunction<PriorRotationConstraintCostFunction, 3, 4>(
                    new PriorRotationConstraintCostFunction(prior_qvec, weight)));
        }

        template<typename T>
        bool operator()(
                const T *const qvec,// qcw
                T *residuals)
                const {
            using Vec3T = Eigen::Matrix<T, 3, 1>;

            Vec3T euler_error;   // 误差的欧拉角

            // 使用 Eigen 的 Quaternion 类来表示四元数，并做逆变换
            const Eigen::Quaternion<T> quat_sfm(qvec[0], -qvec[1], -qvec[2], -qvec[3]);
            // 使用 Eigen 的 Quaternion 类来表示四元数，并做逆变换
            const Eigen::Quaternion<T> quat_prior(
                    T(prior_qvec_(0)),
                    T(-prior_qvec_(1)),
                    T(-prior_qvec_(2)),
                    T(-prior_qvec_(3))
            );
            Eigen::Quaternion<T> quat_error = quat_prior.conjugate() * quat_sfm;
            quat_error.normalize();
            // 将逆变换后的四元数转换为欧拉角
            euler_error = safeEulerAngles(quat_error);
            Eigen::Map<Vec3T> residuals_eigen(residuals);
            residuals_eigen = weight_.cast<T>().cwiseProduct(euler_error);

            return true;
        }
    };


    inline void SetQuaternionManifold(ceres::Problem *problem, double *qvec) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem->SetManifold(qvec, new ceres::QuaternionManifold);
#else
        problem->SetParameterization(qvec, new ceres::QuaternionParameterization);
#endif
    }
    //选择合适的四元数参数化方法
    inline void SetSubsetManifold(int size, const std::vector<int> &constant_params,
                                  ceres::Problem *problem, double *params) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem->SetManifold(params,
                             new ceres::SubsetManifold(size, constant_params));
#else
        problem->SetParameterization(
                params, new ceres::SubsetParameterization(size, constant_params));
#endif
    }

    template<int size>
    inline void SetSphereManifold(ceres::Problem *problem, double *params) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem->SetManifold(params, new ceres::SphereManifold<size>);
#else
        problem->SetParameterization(
                params, new ceres::HomogeneousVectorParameterization(size));
#endif
    }

}// namespace colmap

#endif// COLMAP_SRC_BASE_COST_FUNCTIONS_H_
