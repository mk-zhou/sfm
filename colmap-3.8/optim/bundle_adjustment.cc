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

#include "optim/bundle_adjustment.h"

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/projection.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

#ifdef ENABLE_POSITION_PRIOR
#include "util/numeric.h"
#endif

namespace colmap {

    ////////////////////////////////////////////////////////////////////////////////
    // BundleAdjustmentOptions
    ////////////////////////////////////////////////////////////////////////////////

    ceres::LossFunction *BundleAdjustmentOptions::CreateLossFunction() const {
        ceres::LossFunction *loss_function = nullptr;
        switch (loss_function_type) {
            case LossFunctionType::TRIVIAL:
                loss_function = new ceres::TrivialLoss();
                break;
            case LossFunctionType::SOFT_L1:
                loss_function = new ceres::SoftLOneLoss(loss_function_scale);
                break;
            case LossFunctionType::CAUCHY:
                loss_function = new ceres::CauchyLoss(loss_function_scale);
                break;
        }
        CHECK_NOTNULL(loss_function);
        return loss_function;
    }

    bool BundleAdjustmentOptions::Check() const {
        CHECK_OPTION_GE(loss_function_scale, 0);
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // BundleAdjustmentConfig
    ////////////////////////////////////////////////////////////////////////////////

    BundleAdjustmentConfig::BundleAdjustmentConfig() {}

    size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

    size_t BundleAdjustmentConfig::NumPoints() const {
        return variable_point3D_ids_.size() + constant_point3D_ids_.size();
    }

    size_t BundleAdjustmentConfig::NumConstantCameras() const {
        return constant_camera_ids_.size();
    }

    size_t BundleAdjustmentConfig::NumConstantPoses() const {
        return constant_poses_.size();
    }

    size_t BundleAdjustmentConfig::NumConstantTvecs() const {
        return constant_tvecs_.size();
    }

    size_t BundleAdjustmentConfig::NumVariablePoints() const {
        return variable_point3D_ids_.size();
    }

    size_t BundleAdjustmentConfig::NumConstantPoints() const {
        return constant_point3D_ids_.size();
    }

    size_t BundleAdjustmentConfig::NumResiduals(
            const Reconstruction &reconstruction) const {
        // Count the number of observations for all added images.
        size_t num_observations = 0;
        for (const image_t image_id: image_ids_) {
            num_observations += reconstruction.Image(image_id).NumPoints3D();
        }

        // Count the number of observations for all added 3D points that are not
        // already added as part of the images above.
        // 计算的是一个三维点在不包含在image_ids_中的图像中被观察到的次数
        auto NumObservationsForPoint = [this,
                                        &reconstruction](const point3D_t point3D_id) {
            size_t num_observations_for_point = 0;
            const auto &point3D               = reconstruction.Point3D(point3D_id);
            for (const auto &track_el: point3D.Track().Elements()) {
                if (image_ids_.count(track_el.image_id) == 0) {
                    num_observations_for_point += 1;
                }
            }
            return num_observations_for_point;
        };
        // 遍历variable_point3D_ids_和constant_point3D_ids_中的每个3D点ID，
        // 分别调用NumObservationsForPoint函数计算观测数量，并累加到num_observations中。
        for (const auto point3D_id: variable_point3D_ids_) {
            num_observations += NumObservationsForPoint(point3D_id);
        }
        for (const auto point3D_id: constant_point3D_ids_) {
            num_observations += NumObservationsForPoint(point3D_id);
        }
        // 返回2倍的观测数量作为残差数量。
        //  这是因为在捆绑调整中，每个观测对应两个残差（x和y方向的重投影误差）。
        return 2 * num_observations;
    }

    void BundleAdjustmentConfig::AddImage(const image_t image_id) {
        image_ids_.insert(image_id);
    }

    bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
        return image_ids_.find(image_id) != image_ids_.end();
    }

    void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
        image_ids_.erase(image_id);
    }

    void BundleAdjustmentConfig::SetConstantCamera(const camera_t camera_id) {
        constant_camera_ids_.insert(camera_id);
    }

    void BundleAdjustmentConfig::SetVariableCamera(const camera_t camera_id) {
        constant_camera_ids_.erase(camera_id);
    }

    bool BundleAdjustmentConfig::IsConstantCamera(const camera_t camera_id) const {
        return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
    }

    void BundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
        CHECK(HasImage(image_id));
        //CHECK(!HasConstantTvec(image_id));
        constant_poses_.insert(image_id);
    }

    void BundleAdjustmentConfig::SetVariablePose(const image_t image_id) {
        constant_poses_.erase(image_id);
    }

    bool BundleAdjustmentConfig::HasConstantPose(const image_t image_id) const {
        return constant_poses_.find(image_id) != constant_poses_.end();
    }

    void BundleAdjustmentConfig::SetConstantTvec(const image_t image_id,
                                                 const std::vector<int> &idxs) {
        CHECK_GT(idxs.size(), 0);
        CHECK_LE(idxs.size(), 3);
        CHECK(HasImage(image_id));
        CHECK(!HasConstantPose(image_id));
        CHECK(!VectorContainsDuplicateValues(idxs))
                << "Tvec indices must not contain duplicates";
        constant_tvecs_.emplace(image_id, idxs);
    }

    void BundleAdjustmentConfig::RemoveConstantTvec(const image_t image_id) {
        constant_tvecs_.erase(image_id);
    }

    bool BundleAdjustmentConfig::HasConstantTvec(const image_t image_id) const {
        return constant_tvecs_.find(image_id) != constant_tvecs_.end();
    }

    const std::unordered_set<image_t> &BundleAdjustmentConfig::Images() const {
        return image_ids_;
    }

    const std::unordered_set<point3D_t> &BundleAdjustmentConfig::VariablePoints()
            const {
        return variable_point3D_ids_;
    }

    const std::unordered_set<point3D_t> &BundleAdjustmentConfig::ConstantPoints()
            const {
        return constant_point3D_ids_;
    }

    const std::vector<int> &BundleAdjustmentConfig::ConstantTvec(
            const image_t image_id) const {
        return constant_tvecs_.at(image_id);
    }

    void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
        CHECK(!HasConstantPoint(point3D_id));
        variable_point3D_ids_.insert(point3D_id);
    }

    void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
        CHECK(!HasVariablePoint(point3D_id));
        constant_point3D_ids_.insert(point3D_id);
    }

    bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
        return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
    }

    bool BundleAdjustmentConfig::HasVariablePoint(
            const point3D_t point3D_id) const {
        return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
    }

    bool BundleAdjustmentConfig::HasConstantPoint(
            const point3D_t point3D_id) const {
        return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
    }

    void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
        variable_point3D_ids_.erase(point3D_id);
    }

    void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
        constant_point3D_ids_.erase(point3D_id);
    }

#ifdef ENABLE_POSITION_PRIOR
    void BundleAdjustmentConfig::SetFittingError(
            const double pose_center_robust_fitting_error) {
        pose_center_robust_fitting_error_ = pose_center_robust_fitting_error;
    }
    void BundleAdjustmentConfig::SetPriorPoseWeight(
            const Eigen::Vector3d prior_pose_weight) {
        prior_pose_weight_ = prior_pose_weight;
    }
    void BundleAdjustmentConfig::SetUsagePriorStatus(const bool b_usable_prior) {
        b_usable_prior_ = b_usable_prior;
    }

#endif

    ////////////////////////////////////////////////////////////////////////////////
    // BundleAdjuster
    ////////////////////////////////////////////////////////////////////////////////

    BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions &options,
                                   const BundleAdjustmentConfig &config)
        : options_(options), config_(config) {
        CHECK(options_.Check());
    }

    bool BundleAdjuster::Solve(Reconstruction *reconstruction) {
        // 解决捆绑调整问题的方法

        CHECK_NOTNULL(reconstruction);
        // 检查reconstruction指针是否为空

        CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";
        // 检查是否已经使用过BundleAdjuster，防止多次使用

        // 创建一个新的ceres::Problem对象，并将其通过std::make_unique封装到problem_成员变量中
        // problem_用于存储捆绑调整问题的定义
        problem_ = std::make_unique<ceres::Problem>();

        ceres::LossFunction *loss_function = options_.CreateLossFunction();
        // 根据配置创建损失函数

        SetUp(reconstruction, loss_function);
        // 设置捆绑调整的参数和观测

        if (problem_->NumResiduals() == 0) {
            // 如果问题中没有残差项，则无法进行优化
            return false;
        }

        ceres::Solver::Options solver_options = options_.solver_options;
        // 获取求解器的选项

        const bool has_sparse =
                solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;
        // 检查是否启用稀疏线性代数库

        // 根据图像数量和稀疏性选择线性求解器类型
        const size_t kMaxNumImagesDirectDenseSolver  = 50;
        const size_t kMaxNumImagesDirectSparseSolver = 1000;
        const size_t num_images                      = config_.NumImages();
        if (num_images <= kMaxNumImagesDirectDenseSolver) {
            solver_options.linear_solver_type = ceres::DENSE_SCHUR;
        } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
            solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
        } else {// Indirect sparse (preconditioned CG) solver.
            solver_options.linear_solver_type  = ceres::ITERATIVE_SCHUR;
            solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
        }

        // 根据残差数量确定是否启用多线程
        if (problem_->NumResiduals() <
            options_.min_num_residuals_for_multi_threading) {
            solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
            solver_options.num_linear_solver_threads = 1;
#endif// CERES_VERSION_MAJOR
        } else {
            solver_options.num_threads =
                    GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
            solver_options.num_linear_solver_threads =
                    GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif// CERES_VERSION_MAJOR
        }

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)) << solver_error;
        // 检查求解器选项是否有效

        ceres::Solve(solver_options, problem_.get(), &summary_);
        // 使用Ceres Solver求解捆绑调整问题

        if (solver_options.minimizer_progress_to_stdout) {
            std::cout << "true true true" << std::endl;
            std::cout << std::endl;
        }

        if (options_.print_summary) {
            PrintHeading2("Bundle adjustment report");
            PrintSolverSummary(summary_);


            TearDown(reconstruction);

            return true;
        }
    }
    const ceres::Solver::Summary &BundleAdjuster::Summary() const {
        return summary_;
    }
    // 为ba问题加入图像、三维点并对相机和点进行参数化
    void BundleAdjuster::SetUp(Reconstruction *reconstruction,
                               ceres::LossFunction *loss_function) {
        // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
        // Do not change order of instructions!
        for (const image_t image_id: config_.Images()) {
            AddImageToProblem(image_id, reconstruction, loss_function);
        }
        for (const auto point3D_id: config_.VariablePoints()) {
            AddPointToProblem(point3D_id, reconstruction, loss_function);
        }
        for (const auto point3D_id: config_.ConstantPoints()) {
            AddPointToProblem(point3D_id, reconstruction, loss_function);
        }

        ParameterizeCameras(reconstruction);
        ParameterizePoints(reconstruction);
    }

    void BundleAdjuster::TearDown(Reconstruction *) {
        // Nothing to do
    }
    // 根据图像的姿态类型和相机模型，生成相应的残差函数，
    // 并将其添加到捆绑调整问题中。
    // 这些残差函数用于优化相机姿态和三维点的位置，以最小化重投影误差，从而改善重建结果。
    void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                           Reconstruction *reconstruction,
                                           ceres::LossFunction *loss_function) {
        Image &image   = reconstruction->Image(image_id);
        Camera &camera = reconstruction->Camera(image.CameraId());

        // CostFunction assumes unit quaternions.
        image.NormalizeQvec();

        double *qvec_data          = image.Qvec().data();
        double *tvec_data          = image.Tvec().data();
        double *camera_params_data = camera.ParamsData();
        // 判断是否是恒定的姿态
        const bool constant_pose =
                !options_.refine_extrinsics || config_.HasConstantPose(image_id);

#ifdef ENABLE_POSITION_PRIOR
        if (config_.GetUsagePriorStatus()) {
            const Eigen::Vector3d pose_center_constraint =
                    ProjectionCenterFromPose(image.QvecPrior(), image.TvecPrior());

            // TODO: need to auto adjust XYZ weight
            const Eigen::Vector3d xyz_weight = config_.GetPriorPoseWeight();
            // Add the cost functor (distance from Pose prior to the SfM_Data Pose
            // center)
            ceres::CostFunction *cost_function      = nullptr;
            double pose_center_robust_fitting_error = config_.GetFittingError();
            cost_function                           = PoseCenterConstraintCostFunction::Create(
                    pose_center_constraint, xyz_weight);

            problem_->AddResidualBlock(
                    cost_function,
                    new ceres::HuberLoss(Square(pose_center_robust_fitting_error)),
                    qvec_data, tvec_data);
            // 车体的q，t（snapshot类）
            // 相机相对于车体的q，t（camera类）
            // image类->snapshot和camera
        }
#endif

        // Add residuals to bundle adjustment problem.
        size_t num_observations = 0;
        // 遍历该图像上的所有二维点
        std::cout << "image.Points2D().size():" << image.Points2D().size() << std::endl;
        for (const Point2D &point2D: image.Points2D()) {
            // 没有三维点的话处理下一个点
            if (!point2D.HasPoint3D()) {
                continue;
            }
            std::cout << "point2D.HasPoint3D()" << std::endl;
            num_observations += 1;
            // 三维点对应的二维点数量+1
            point3D_num_observations_[point2D.Point3DId()] += 1;
            Point3D &point3D = reconstruction->Point3D(point2D.Point3DId());
            assert(point3D.Track().Length() > 1);

            ceres::CostFunction *cost_function = nullptr;
            // 如果相机位姿恒定，采用ConstantPoseCostFunction这一损失函数
            if (constant_pose) {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                }

                problem_->AddResidualBlock(cost_function, loss_function,
                                           point3D.XYZ().data(), camera_params_data);
            }
            // 如果相机位姿不恒定，采用CostFunction这一损失函数
            else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                           \
    case CameraModel::kModelId:                                                  \
        cost_function =                                                          \
                BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                }

                std::cout << "camera_params_data.size()=" << *camera_params_data << std::endl;
                problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                           tvec_data, point3D.XYZ().data(),
                                           camera_params_data);
            }
        }

        if (num_observations > 0) {
            camera_ids_.insert(image.CameraId());

            // Set pose parameterization.
            if (!constant_pose) {
                SetQuaternionManifold(problem_.get(), qvec_data);
                if (config_.HasConstantTvec(image_id)) {
                    const std::vector<int> &constant_tvec_idxs =
                            config_.ConstantTvec(image_id);
                    SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
                }
            }
        }
    }

    void BundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                           Reconstruction *reconstruction,
                                           ceres::LossFunction *loss_function) {
        // Get the point3D object.
        Point3D &point3D = reconstruction->Point3D(point3D_id);

        // Is 3D point already fully contained in the problem? I.e. its entire track
        // is contained in `variable_image_ids`, `constant_image_ids`,
        // `constant_x_image_ids`.
        if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
            return;
        }

        // Add the point to the problem.
        for (const auto &track_el: point3D.Track().Elements()) {
            // Skip observations that were already added in `FillImages`.
            if (config_.HasImage(track_el.image_id)) {
                continue;
            }

            // Increment the number of observations for the point.
            point3D_num_observations_[point3D_id] += 1;

            // Get the image object.
            Image &image           = reconstruction->Image(track_el.image_id);
            Camera &camera         = reconstruction->Camera(image.CameraId());
            const Point2D &point2D = image.Point2D(track_el.point2D_idx);

            // We do not want to refine the camera of images that are not
            // part of `constant_image_ids_`, `constant_image_ids_`,
            // `constant_x_image_ids_`.
            if (camera_ids_.count(image.CameraId()) == 0) {
                camera_ids_.insert(image.CameraId());
                config_.SetConstantCamera(image.CameraId());
            }

            ceres::CostFunction *cost_function = nullptr;

            switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        break;

                CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
            }
            // Add the residual block to the problem.
            problem_->AddResidualBlock(cost_function, loss_function,
                                       point3D.XYZ().data(), camera.ParamsData());
        }
    }

    void BundleAdjuster::ParameterizeCameras(Reconstruction *reconstruction) {
        // 有一个需要约束的话就不是恒定相机，可以通过设置选项让所有相机都变为恒定相机
        bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
        constant_camera = true;
        for (auto it = camera_ids_.begin(); it != camera_ids_.end(); ++it) {
            auto camera_id = *it;
            Camera &camera = reconstruction->Camera(camera_id);
            std::cout << std::endl;
            if (constant_camera || config_.IsConstantCamera(camera_id)) {
                problem_->SetParameterBlockConstant(camera.ParamsData());
                continue;
            } else {
                std::vector<int> const_camera_params;
                // 如果某项不需要优化，则将其设定为常数
                if (!options_.refine_focal_length) {
                    const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                    const_camera_params.insert(const_camera_params.end(),
                                               params_idxs.begin(), params_idxs.end());
                }
                if (!options_.refine_principal_point) {
                    const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                    const_camera_params.insert(const_camera_params.end(),
                                               params_idxs.begin(), params_idxs.end());
                }
                if (!options_.refine_extra_params) {
                    const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                    const_camera_params.insert(const_camera_params.end(),
                                               params_idxs.begin(), params_idxs.end());
                }
                // 如果const_camera_params的大小大于0
                if (const_camera_params.size() > 0) {
                    std::cout << "const_camera_params.size():" << const_camera_params.size() << std::endl;
                    SetSubsetManifold(static_cast<int>(camera.NumParams()),
                                      const_camera_params, problem_.get(),
                                      camera.ParamsData());
                    // 调用SetSubsetManifold函数，将相机参数中的指定索引列表const_camera_params标记为常量
                }
            }
        }
    }
    void BundleAdjuster::ParameterizePoints(Reconstruction *reconstruction) {
        // 点参数化函数，用于标记点参数为常量

        for (const auto elem: point3D_num_observations_) {
            // 遍历点的观测数量信息

            Point3D &point3D = reconstruction->Point3D(elem.first);
            // 获取点的三维对象

            if (point3D.Track().Length() > elem.second) {
                // 如果点的轨迹长度大于观测数量

                problem_->SetParameterBlockConstant(point3D.XYZ().data());
                // 将点的参数块标记为常量
            }
        }

        for (const point3D_t point3D_id: config_.ConstantPoints()) {
            // 遍历配置中的常量点

            Point3D &point3D = reconstruction->Point3D(point3D_id);
            // 获取点的三维对象

            problem_->SetParameterBlockConstant(point3D.XYZ().data());
            // 将点的参数块标记为常量
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // ParallelBundleAdjuster
    ////////////////////////////////////////////////////////////////////////////////

    bool ParallelBundleAdjuster::Options::Check() const {
        CHECK_OPTION_GE(max_num_iterations, 0);
        return true;
    }

    ParallelBundleAdjuster::ParallelBundleAdjuster(
            const Options &options, const BundleAdjustmentOptions &ba_options,
            const BundleAdjustmentConfig &config)
        : options_(options),
          ba_options_(ba_options),
          config_(config),
          num_measurements_(0) {
        CHECK(options_.Check());
        CHECK(ba_options_.Check());
        CHECK_EQ(config_.NumConstantCameras(), 0)
                << "PBA does not allow to set individual cameras constant";
        CHECK_EQ(config_.NumConstantPoses(), 0)
                << "PBA does not allow to set individual translational elements constant";
        CHECK_EQ(config_.NumConstantTvecs(), 0)
                << "PBA does not allow to set individual translational elements constant";
        CHECK(config_.NumVariablePoints() == 0 && config_.NumConstantPoints() == 0)
                << "PBA does not allow to parameterize individual 3D points";
    }

    bool ParallelBundleAdjuster::Solve(Reconstruction *reconstruction) {
        CHECK_NOTNULL(reconstruction);
        CHECK_EQ(num_measurements_, 0)
                << "Cannot use the same ParallelBundleAdjuster multiple times";
        CHECK(!ba_options_.refine_principal_point);
        CHECK_EQ(ba_options_.refine_focal_length, ba_options_.refine_extra_params);

        SetUp(reconstruction);

        const int num_residuals = static_cast<int>(2 * measurements_.size());

        size_t num_threads = options_.num_threads;
        if (num_residuals < options_.min_num_residuals_for_multi_threading) {
            num_threads = 1;
        }

        pba::ParallelBA::DeviceT device;
        const int kMaxNumResidualsFloat = 100 * 1000;
        if (num_residuals > kMaxNumResidualsFloat) {
            // The threshold for using double precision is empirically chosen and
            // ensures that the system can be reliable solved.
            device = pba::ParallelBA::PBA_CPU_DOUBLE;
        } else {
            if (options_.gpu_index < 0) {
                device = pba::ParallelBA::PBA_CUDA_DEVICE_DEFAULT;
            } else {
                device = static_cast<pba::ParallelBA::DeviceT>(
                        pba::ParallelBA::PBA_CUDA_DEVICE0 + options_.gpu_index);
            }
        }

        pba::ParallelBA pba(device, num_threads);

        pba.SetNextBundleMode(pba::ParallelBA::BUNDLE_FULL);
        pba.EnableRadialDistortion(pba::ParallelBA::PBA_PROJECTION_DISTORTION);
        pba.SetFixedIntrinsics(!ba_options_.refine_focal_length &&
                               !ba_options_.refine_extra_params);

        pba::ConfigBA *pba_config = pba.GetInternalConfig();
        pba_config->__lm_delta_threshold /= 100.0f;
        pba_config->__lm_gradient_threshold /= 100.0f;
        pba_config->__lm_mse_threshold = 0.0f;
        pba_config->__cg_min_iteration = 10;
        pba_config->__verbose_level    = 2;
        pba_config->__lm_max_iteration = options_.max_num_iterations;

        pba.SetCameraData(cameras_.size(), cameras_.data());
        pba.SetPointData(points3D_.size(), points3D_.data());
        pba.SetProjection(measurements_.size(), measurements_.data(),
                          point3D_idxs_.data(), camera_idxs_.data());

        Timer timer;
        timer.Start();
        pba.RunBundleAdjustment();
        timer.Pause();

        // Compose Ceres solver summary from PBA options.
        summary_.num_residuals_reduced = num_residuals;
        summary_.num_effective_parameters_reduced =
                static_cast<int>(8 * config_.NumImages() -
                                 2 * config_.NumConstantCameras() + 3 * points3D_.size());
        summary_.num_successful_steps = pba_config->GetIterationsLM() + 1;
        summary_.termination_type     = ceres::TerminationType::USER_SUCCESS;
        summary_.initial_cost =
                pba_config->GetInitialMSE() * summary_.num_residuals_reduced / 4;
        summary_.final_cost =
                pba_config->GetFinalMSE() * summary_.num_residuals_reduced / 4;
        summary_.total_time_in_seconds = timer.ElapsedSeconds();

        TearDown(reconstruction);

        if (options_.print_summary) {
            PrintHeading2("Bundle adjustment report");
            PrintSolverSummary(summary_);
        }

        return true;
    }

    const ceres::Solver::Summary &ParallelBundleAdjuster::Summary() const {
        return summary_;
    }

    bool ParallelBundleAdjuster::IsSupported(const BundleAdjustmentOptions &options,
                                             const Reconstruction &reconstruction) {
        if (options.refine_principal_point ||
            options.refine_focal_length != options.refine_extra_params) {
            return false;
        }

        // Check that all cameras are SIMPLE_RADIAL and that no intrinsics are shared.
        std::set<camera_t> camera_ids;
        for (const auto &image: reconstruction.Images()) {
            if (image.second.IsRegistered()) {
                if (camera_ids.count(image.second.CameraId()) != 0 ||
                    reconstruction.Camera(image.second.CameraId()).ModelId() !=
                            SimpleRadialCameraModel::model_id) {
                    return false;
                }
                camera_ids.insert(image.second.CameraId());
            }
        }
        return true;
    }

    void ParallelBundleAdjuster::SetUp(Reconstruction *reconstruction) {
        // Important: PBA requires the track of 3D points to be stored
        // contiguously, i.e. the point3D_idxs_ vector contains consecutive indices.
        cameras_.reserve(config_.NumImages());
        camera_ids_.reserve(config_.NumImages());
        ordered_image_ids_.reserve(config_.NumImages());
        image_id_to_camera_idx_.reserve(config_.NumImages());
        AddImagesToProblem(reconstruction);
        AddPointsToProblem(reconstruction);
    }

    void ParallelBundleAdjuster::TearDown(Reconstruction *reconstruction) {
        for (size_t i = 0; i < cameras_.size(); ++i) {
            const image_t image_id         = ordered_image_ids_[i];
            const pba::CameraT &pba_camera = cameras_[i];

            // Note: Do not use PBA's quaternion methods as they seem to lead to
            // numerical instability or other issues.
            Image &image = reconstruction->Image(image_id);
            Eigen::Matrix3d rotation_matrix;
            pba_camera.GetMatrixRotation(rotation_matrix.data());
            pba_camera.GetTranslation(image.Tvec().data());
            image.Qvec() = RotationMatrixToQuaternion(rotation_matrix.transpose());

            Camera &camera   = reconstruction->Camera(image.CameraId());
            camera.Params(0) = pba_camera.GetFocalLength();
            camera.Params(3) = pba_camera.GetProjectionDistortion();
        }

        for (size_t i = 0; i < points3D_.size(); ++i) {
            Point3D &point3D = reconstruction->Point3D(ordered_point3D_ids_[i]);
            points3D_[i].GetPoint(point3D.XYZ().data());
        }
    }

    void ParallelBundleAdjuster::AddImagesToProblem(
            Reconstruction *reconstruction) {
        for (const image_t image_id: config_.Images()) {
            const Image &image = reconstruction->Image(image_id);
            CHECK_EQ(camera_ids_.count(image.CameraId()), 0)
                    << "PBA does not support shared intrinsics";

            const Camera &camera = reconstruction->Camera(image.CameraId());
            CHECK_EQ(camera.ModelId(), SimpleRadialCameraModel::model_id)
                    << "PBA only supports the SIMPLE_RADIAL camera model";

            // Note: Do not use PBA's quaternion methods as they seem to lead to
            // numerical instability or other issues.
            const Eigen::Matrix3d rotation_matrix =
                    QuaternionToRotationMatrix(image.Qvec()).transpose();

            pba::CameraT pba_camera;
            pba_camera.SetFocalLength(camera.Params(0));
            pba_camera.SetProjectionDistortion(camera.Params(3));
            pba_camera.SetMatrixRotation(rotation_matrix.data());
            pba_camera.SetTranslation(image.Tvec().data());

            CHECK(!config_.HasConstantTvec(image_id))
                    << "PBA cannot fix partial extrinsics";
            if (!ba_options_.refine_extrinsics || config_.HasConstantPose(image_id)) {
                CHECK(config_.IsConstantCamera(image.CameraId()))
                        << "PBA cannot fix extrinsics only";
                pba_camera.SetConstantCamera();
            } else if (config_.IsConstantCamera(image.CameraId())) {
                pba_camera.SetFixedIntrinsic();
            } else {
                pba_camera.SetVariableCamera();
            }

            num_measurements_ += image.NumPoints3D();
            cameras_.push_back(pba_camera);
            camera_ids_.insert(image.CameraId());
            ordered_image_ids_.push_back(image_id);
            image_id_to_camera_idx_.emplace(image_id,
                                            static_cast<int>(cameras_.size()) - 1);

            for (const Point2D &point2D: image.Points2D()) {
                if (point2D.HasPoint3D()) {
                    point3D_ids_.insert(point2D.Point3DId());
                }
            }
        }
    }

    void ParallelBundleAdjuster::AddPointsToProblem(
            Reconstruction *reconstruction) {
        points3D_.resize(point3D_ids_.size());
        ordered_point3D_ids_.resize(point3D_ids_.size());
        measurements_.resize(num_measurements_);
        camera_idxs_.resize(num_measurements_);
        point3D_idxs_.resize(num_measurements_);

        int point3D_idx        = 0;
        size_t measurement_idx = 0;

        for (const auto point3D_id: point3D_ids_) {
            const Point3D &point3D = reconstruction->Point3D(point3D_id);
            points3D_[point3D_idx].SetPoint(point3D.XYZ().data());
            ordered_point3D_ids_[point3D_idx] = point3D_id;

            for (const auto track_el: point3D.Track().Elements()) {
                if (image_id_to_camera_idx_.count(track_el.image_id) > 0) {
                    const Image &image     = reconstruction->Image(track_el.image_id);
                    const Camera &camera   = reconstruction->Camera(image.CameraId());
                    const Point2D &point2D = image.Point2D(track_el.point2D_idx);
                    measurements_[measurement_idx].SetPoint2D(
                            point2D.X() - camera.Params(1), point2D.Y() - camera.Params(2));
                    camera_idxs_[measurement_idx] =
                            image_id_to_camera_idx_.at(track_el.image_id);
                    point3D_idxs_[measurement_idx] = point3D_idx;
                    measurement_idx += 1;
                }
            }
            point3D_idx += 1;
        }

        CHECK_EQ(point3D_idx, points3D_.size());
        CHECK_EQ(measurement_idx, measurements_.size());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // RigBundleAdjuster
    ////////////////////////////////////////////////////////////////////////////////

    RigBundleAdjuster::RigBundleAdjuster(const BundleAdjustmentOptions &options,
                                         const Options &rig_options,
                                         const BundleAdjustmentConfig &config)
        : BundleAdjuster(options, config), rig_options_(rig_options) {}
    // 构造函数，初始化RigBundleAdjuster对象

    bool RigBundleAdjuster::Solve(Reconstruction *reconstruction,
                                  std::vector<CameraRig> *camera_rigs) {
        // 解决带有相机组的捆绑调整问题
        std::cout << "solve" << std::endl;
        CHECK_NOTNULL(reconstruction);
        CHECK_NOTNULL(camera_rigs);
        CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";
        // 检查输入指针是否为空，以及是否已经使用过BundleAdjuster

        // 检查提供的相机组的有效性
        std::unordered_set<camera_t> rig_camera_ids;
        for (auto &camera_rig: *camera_rigs) {
            camera_rig.Check(*reconstruction);
            // 检查相机组的有效性

            for (const auto &camera_id: camera_rig.GetCameraIds()) {
                CHECK_EQ(rig_camera_ids.count(camera_id), 0)
                        << "Camera must not be part of multiple camera rigs";
                //std::cout << "camera_id:" << camera_id << std::endl;
                rig_camera_ids.insert(camera_id);
            }
            // 检查相机是否属于多个相机组

            for (const auto &snapshot: camera_rig.Snapshots()) {
                for (const auto &image_id: snapshot) {
                    CHECK_EQ(image_id_to_camera_rig_.count(image_id), 0)
                            << "Image must not be part of multiple camera rigs";
                    image_id_to_camera_rig_.emplace(image_id, &camera_rig);
                    //std::cout << "image_id:" << image_id << std::endl;
                }
            }
            // 检查图像是否属于多个相机组
        }
        problem_ = std::make_unique<ceres::Problem>();
        // 创建Ceres Solver问题对象

        ceres::LossFunction *loss_function = options_.CreateLossFunction();
        // 创建损失函数

        SetUp(reconstruction, camera_rigs, loss_function);
        // 设置捆绑调整的参数和观测
        if (problem_->NumResiduals() == 0) {
            // 如果问题中没有残差项，则无法进行优化
            return false;
        }

        ceres::Solver::Options solver_options = options_.solver_options;
        // 获取求解器的选项

        const bool has_sparse =
                solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;
        /*if (has_sparse) {
            std::cout << "has sparse" << std::endl;
        } else {
            std::cout << "no sparse" << std::endl;
        }*/
        // 检查是否启用稀疏线性代数库

        // 根据图像数量和稀疏性选择线性求解器类型
        const size_t kMaxNumImagesDirectDenseSolver  = 50;
        const size_t kMaxNumImagesDirectSparseSolver = 1000;
        const size_t num_images                      = config_.NumImages();
        if (num_images <= kMaxNumImagesDirectDenseSolver) {
            //std::cout << "using dense solver";
            solver_options.linear_solver_type = ceres::DENSE_SCHUR;
        } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
            //std::cout << "using sparse solver";
            solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
        } else {// Indirect sparse (preconditioned CG) solver.
            //std::cout << "using sparse solver and schue_jacobi";
            //solver_options.linear_solver_type  = ceres::ITERATIVE_SCHUR;
            solver_options.linear_solver_type  = ceres::SPARSE_SCHUR;
            solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
        }
        solver_options.num_threads =
                GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads =
                GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif// CERES_VERSION_MAJOR

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)) << solver_error;
        // 检查求解器选项是否有效
        ceres::Solve(solver_options, problem_.get(), &summary_);
        // 使用Ceres Solver求解捆绑调整问题
        if (options_.print_summary) {
            PrintHeading2("Rig Bundle adjustment report");
            PrintSolverSummary(summary_);
        }
        // 更新每个在
        TearDown(reconstruction, *camera_rigs);

        return true;
    }

    void RigBundleAdjuster::WriteCarPoseAndExtri(
            Reconstruction *reconstruction, std::vector<CameraRig> *camera_rigs,
            const std::string &output_path) {
        ComputeCameraRigPoses(*reconstruction, *camera_rigs);
        std::string output_carposes_txt = output_path + "/car_poses.txt";
        std::ofstream outfile(output_carposes_txt);
        if (outfile.is_open()) {
            for (auto &camera_rig: *camera_rigs) {
                int num_of_snapshots = camera_rig.NumSnapshots();
                for (auto &snapshot: camera_rig.Snapshots()) {
                    for (auto image_id: snapshot) {
                        const auto &image    = reconstruction->Image(image_id);
                        const auto camera_id = image.CameraId();
                        if (camera_id != camera_rig.RefCameraId()) {
                            continue;
                        }
                        std::string image_id_str  = std::to_string(image_id);
                        std::string camera_id_str = std::to_string(image.CameraId());
                        std::string image_name    = image.Name();// 获取指针
                        Eigen::Vector4d *qvec_ptr = image_id_to_rig_qvec_[image_id];
                        std::string qw_str =
                                std::to_string((*qvec_ptr)(0));// 将每个元素转换为字符串
                        std::string qx_str   = std::to_string((*qvec_ptr)(1));
                        std::string qy_str   = std::to_string((*qvec_ptr)(2));
                        std::string qz_str   = std::to_string((*qvec_ptr)(3));
                        std::string qvec_str = qw_str + " " + qx_str + " " + qy_str + " " +
                                               qz_str + " ";                        // 将字符串连接起来
                        Eigen::Vector3d *tvec_ptr = image_id_to_rig_tvec_[image_id];// 获取指针
                        std::string x_str =
                                std::to_string((*tvec_ptr)(0));// 将每个元素转换为字符串
                        std::string y_str = std::to_string((*tvec_ptr)(1));
                        std::string z_str = std::to_string((*tvec_ptr)(2));
                        std::string tvec_str =
                                x_str + " " + y_str + " " + z_str + " ";// 将字符串连接起来
                        std::string output_str = image_id_str + " " + qvec_str + tvec_str +
                                                 camera_id_str + " " + image_name + "\n\n";
                        outfile << output_str;
                    }
                }
            }

            std::cout << "car poses xieru chenggong";
        } else {
            std::cerr << "car poses wufa xieru wenjian";
        }
        std::string output_camposes_txt = output_path + "/cam_poses.txt";
        // int num_of_cameras = reconstruction->NumCameras();
        std::ofstream cam_outfile(output_camposes_txt);
        if (cam_outfile.is_open()) {
            for (auto &camera_rig: *camera_rigs) {
                int camera_num = camera_rig.NumCameras();
                for (auto camera_id: camera_rig.GetCameraIds()) {
                    std::cout << camera_id << std::endl;

                    std::string image_id_str  = std::to_string(camera_id);
                    std::string camera_id_str = std::to_string(camera_id);
                    std::string image_name    = std::to_string(camera_id) + ".jpg";
                    Eigen::Vector4d qvec_ptr =
                            camera_rig.RelativeQvec(camera_id);// 获取指针
                    std::string qw_str =
                            std::to_string(qvec_ptr[0]);// 将每个元素转换为字符串
                    std::string qx_str   = std::to_string(qvec_ptr[1]);
                    std::string qy_str   = std::to_string(qvec_ptr[2]);
                    std::string qz_str   = std::to_string(qvec_ptr[3]);
                    std::string qvec_str = qw_str + " " + qx_str + " " + qy_str + " " +
                                           qz_str + " ";// 将字符串连接起来
                    Eigen::Vector3d tvec_ptr =
                            camera_rig.RelativeTvec(camera_id);// 获取指针
                    std::string x_str =
                            std::to_string(tvec_ptr[0]);// 将每个元素转换为字符串
                    std::string y_str = std::to_string(tvec_ptr[1]);
                    std::string z_str = std::to_string(tvec_ptr[2]);
                    std::string tvec_str =
                            x_str + " " + y_str + " " + z_str + " ";// 将字符串连接起来
                    std::string output_str = image_id_str + " " + qvec_str + tvec_str +
                                             camera_id_str + " " + image_name + "\n\n";
                    cam_outfile << output_str;
                }
            }

            std::cout << "cam poses xieru chenggong";
        } else {
            std::cerr << "cam poses wufa xieru wenjian";
        }
    }

    void RigBundleAdjuster::SetUp(Reconstruction *reconstruction,
                                  std::vector<CameraRig> *camera_rigs,
                                  ceres::LossFunction *loss_function) {
        ComputeCameraRigPoses(*reconstruction, *camera_rigs);
        for (const image_t image_id: config_.Images()) {
            //std::cout << "image_id:" << image_id << std::endl;
            AddImageToProblem(image_id, reconstruction, camera_rigs, loss_function);
        }
        for (const auto point3D_id: config_.VariablePoints()) {
            AddPointToProblem(point3D_id, reconstruction, loss_function);
        }
        for (const auto point3D_id: config_.ConstantPoints()) {
            AddPointToProblem(point3D_id, reconstruction, loss_function);
        }

        ParameterizeCameras(reconstruction);
        ParameterizePoints(reconstruction);
        ParameterizeCameraRigs(reconstruction);
    }

    // 得到每个有对应rig的图片的位姿
    void RigBundleAdjuster::TearDown(Reconstruction *reconstruction,
                                     const std::vector<CameraRig> &camera_rigs) {
        // 所有图片的位姿都通过 相机组位姿 和 相对位姿 进行了解算
        //std::cout << "size of image_id_to_rig:" << image_id_to_camera_rig_.size() << std::endl;
        for (const auto &elem: image_id_to_camera_rig_) {
            // 遍历图像ID到相机组的映射
            //std::cout << "image_id:" << elem.first << std::endl;
            if (!reconstruction->IsImageRegistered(elem.first)) { continue; }
            const auto image_id    = elem.first;
            const auto &camera_rig = *elem.second;
            // 获取图像ID和相机组的引用

            auto &image = reconstruction->Image(image_id);
            // 获取重建中对应图像的引用

            // ceres优化的是image_id_to_rig_和Relative里的量
            // 所以需要一个TearDown把每张图片的信息更新
            // 而常规BA直接优化图片的位姿，所以不需要TearDown

            // 问题
            // 同一时刻的图片对应的相机组位姿是否一样 （）
            // 同一类别的图片对应的相对位姿是否一样（一致，因为RelativePose是靠CameraId区分的）
            ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                             *image_id_to_rig_tvec_.at(image_id),
                             camera_rig.RelativeQvec(image.CameraId()),
                             camera_rig.RelativeTvec(image.CameraId()), &image.Qvec(),
                             &image.Tvec());
            // 得到世界坐标系相对于图片的位姿
        }
    }

    void RigBundleAdjuster::AddImageToProblem(const image_t image_id,
                                              Reconstruction *reconstruction,
                                              std::vector<CameraRig> *camera_rigs,
                                              ceres::LossFunction *loss_function) {
        // 获取最大重投影误差的平方
        const double max_squared_reproj_error =
                rig_options_.max_reproj_error * rig_options_.max_reproj_error;

        // 获取图像和相机对象的引用
        Image &image   = reconstruction->Image(image_id);
        Camera &camera = reconstruction->Camera(image.CameraId());

        // 检查图像是否具有常量姿态和平移向量
        const bool constant_pose = config_.HasConstantPose(image_id);
        const bool constant_tvec = config_.HasConstantTvec(image_id);

        // 初始化指针变量
        double *qvec_data                 = nullptr;
        double *tvec_data                 = nullptr;
        double *rig_qvec_data             = nullptr;
        double *rig_tvec_data             = nullptr;
        double *camera_params_data        = camera.ParamsData();
        CameraRig *camera_rig             = nullptr;
        Eigen::Matrix3x4d rig_proj_matrix = Eigen::Matrix3x4d::Zero();


        // 检查图像是否属于相机组
        if (image_id_to_camera_rig_.count(image_id) > 0) {
            // 如果图像属于相机组，则图像不应具有常量姿态和平移向量
            /*CHECK(!constant_pose)
                    << "Images contained in a camera rig must not have constant pose";
            CHECK(!constant_tvec)
                    << "Images contained in a camera rig must not have constant tvec";*/

            // 获取图像ID对应的相机组和相关的姿态数据
            camera_rig = image_id_to_camera_rig_.at(image_id);
            // 相机组的绝对位姿
            rig_qvec_data = image_id_to_rig_qvec_.at(image_id)->data();
            rig_tvec_data = image_id_to_rig_tvec_.at(image_id)->data();
            // 在计算rig_pose时，同一时刻图片都有同一个rig_pose
            // 相机相对相机组的位姿
            qvec_data = camera_rig->RelativeQvec(image.CameraId()).data();
            tvec_data = camera_rig->RelativeTvec(image.CameraId()).data();


            // 通过世界相对rig位姿和图片相对rig位姿
            // 计算得到世界相对图片位姿
            Eigen::Vector4d rig_concat_qvec;
            Eigen::Vector3d rig_concat_tvec;
            ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                             *image_id_to_rig_tvec_.at(image_id),
                             camera_rig->RelativeQvec(image.CameraId()),
                             camera_rig->RelativeTvec(image.CameraId()),
                             &rig_concat_qvec, &rig_concat_tvec);
            // 通过将四元数变换成旋转矩阵
            // 加上位移合成投影矩阵
            rig_proj_matrix = ComposeProjectionMatrix(rig_concat_qvec, rig_concat_tvec);
        } else {
            // 如果图像不属于相机组，则将其四元数归一化
            image.NormalizeQvec();
            qvec_data = image.Qvec().data();
            tvec_data = image.Tvec().data();
        }

        // 收集相机ID以进行最终参数化
        CHECK(image.HasCamera());
        // std::cout<<"adding camera_ids_:"<<image.CameraId()<<std::endl;
        camera_ids_.insert(image.CameraId());

        // 当前图像添加的观测数量
        size_t num_observations = 0;

        for (const Point2D &point2D: image.Points2D()) {
            if (!point2D.HasPoint3D()) {
                continue;
            }

            Point3D &point3D = reconstruction->Point3D(point2D.Point3DId());
            assert(point3D.Track().Length() > 1);

            // 如果图像属于相机组且重投影误差超过阈值，则跳过该观测
            if (camera_rig != nullptr &&
                CalculateSquaredReprojectionError(point2D.XY(), point3D.XYZ(),
                                                  rig_proj_matrix,
                                                  camera) > max_squared_reproj_error) {
                //std::cout << "have too large reprojection error" << CalculateSquaredReprojectionError(point2D.XY(), point3D.XYZ(), rig_proj_matrix, camera) << std::endl;
                continue;
            }

            // 该3d点满足要求，记录其编号
            num_observations += 1;
            point3D_num_observations_[point2D.Point3DId()] += 1;

            ceres::CostFunction *cost_function = nullptr;

            if (constant_pose && camera_rig != nullptr) {
                std::cout << "constant pose " << image_id << std::endl;
                // 如果图像具有常量姿态
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                }
                // std::cout<<"adding at 2";
                problem_->AddResidualBlock(cost_function, loss_function,
                                           point3D.XYZ().data(), camera_params_data);
                continue;
            }

            // 如果图像不属于相机组
            if (camera_rig == nullptr) {
                // 如果图像具有常量姿态
                if (constant_pose) {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    // std::cout<<"adding at 2";
                    problem_->AddResidualBlock(cost_function, loss_function,
                                               point3D.XYZ().data(), camera_params_data);
                }
                // 如果图像没有常量姿态
                else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                           \
    case CameraModel::kModelId:                                                  \
        cost_function =                                                          \
                BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
        break;

                        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                    // std::cout<<"adding at 2";
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                               tvec_data, point3D.XYZ().data(),
                                               camera_params_data);
                }
            }
            // 如果图像属于相机组
            else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                              \
    case CameraModel::kModelId:                                                     \
        cost_function =                                                             \
                RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
                                                                                    \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                }
                // std::cout<<image.CameraId()<<" adding at 3"<<std::endl;
                problem_->AddResidualBlock(cost_function, loss_function, rig_qvec_data,
                                           rig_tvec_data, qvec_data, tvec_data,
                                           point3D.XYZ().data(), camera_params_data);
            }
        }
        // 如果观测数量大于0
        if (num_observations > 0) {

            if (camera_rig != nullptr && !constant_pose) {
                parameterized_qvec_data_.insert(qvec_data);
                parameterized_qvec_data_.insert(rig_qvec_data);

                // 如果相对姿态的优化被禁用或者当前图像是参考相机，将相机姿态参数块设置为常量，避免过度参数化
                if (!rig_options_.refine_relative_poses ||
                    image.CameraId() == camera_rig->RefCameraId()) {
                    problem_->SetParameterBlockConstant(qvec_data);
                }
                problem_->SetParameterBlockConstant(tvec_data);
            }

            // 设置姿态参数化
            if (!constant_pose && constant_tvec) {
                const std::vector<int> &constant_tvec_idxs =
                        config_.ConstantTvec(image_id);
                SetSubsetManifold(3, constant_tvec_idxs, problem_.get(), tvec_data);
            }
        }
    }

    void RigBundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                              Reconstruction *reconstruction,
                                              ceres::LossFunction *loss_function) {
        Point3D &point3D = reconstruction->Point3D(point3D_id);

        // Is 3D point already fully contained in the problem? I.e. its entire track
        // is contained in `variable_image_ids`, `constant_image_ids`,
        // `constant_x_image_ids`.
        // 当这个3d点被观测的次数等于它track的长度
        // 例如一个3d点被3张图观测到，它最多被添加到residual里面3次
        // 就不再添加这个点
        // 避免重复添加
        if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
            return;
        }
        // 如果被3张图片观测到，则将他加入残差3次
        for (const auto &track_el: point3D.Track().Elements()) {
            // Skip observations that were already added in `AddImageToProblem`.
            // 图片已被添加，说明这个三维点在那时候已经被添加一次了
            // 如果想被添加，那么一定要有另一张未被添加的图片也包含它
            if (config_.HasImage(track_el.image_id)) {
                continue;
            }

            point3D_num_observations_[point3D_id] += 1;
            // 引入3d点对应的图片，相机，以及对应图片上的2d点.
            Image &image           = reconstruction->Image(track_el.image_id);
            Camera &camera         = reconstruction->Camera(image.CameraId());
            const Point2D &point2D = image.Point2D(track_el.point2D_idx);

            // We do not want to refine the camera of images that are not
            // part of `constant_image_ids_`, `constant_image_ids_`,
            // `constant_x_image_ids_`.
            // 如果这个图片对应的相机还没有被添加到problem里面（此处是第一次）
            // 把这个相机设定为常量
            if (camera_ids_.count(image.CameraId()) == 0) {
                camera_ids_.insert(image.CameraId());
                config_.SetConstantCamera(image.CameraId());
            }

            ceres::CostFunction *cost_function = nullptr;

            // 增加3d点的坐标到residual里面，保持图片位姿和2d点坐标不变
            switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        problem_->AddResidualBlock(cost_function, loss_function,               \
                                   point3D.XYZ().data(), camera.ParamsData()); \
        break;

                CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
            }
        }
    }

    void RigBundleAdjuster::ComputeCameraRigPoses(
            const Reconstruction &reconstruction,
            std::vector<CameraRig> &camera_rigs) {
        camera_rig_qvecs_.reserve(camera_rigs.size());
        camera_rig_tvecs_.reserve(camera_rigs.size());
        for (auto &camera_rig: camera_rigs) {// 遍历每一个相机组
            // 在camera_rig_qvecs_容器的末尾构造一个对象
            camera_rig.ComputeRelativePoses(reconstruction);
            camera_rig_qvecs_.emplace_back();
            camera_rig_tvecs_.emplace_back();
            // 创建引用rig_qvecs指向camera_rig_qvecs_容器的最后一个对象
            auto &rig_qvecs = camera_rig_qvecs_.back();
            auto &rig_tvecs = camera_rig_tvecs_.back();
            // 使用resize()函数将rig_qvecs的大小调整为camera_rig.NumSnapshots()
            rig_qvecs.resize(camera_rig.NumSnapshots());
            rig_tvecs.resize(camera_rig.NumSnapshots());
            // 对每一个snapshot里面所有图片计算得到参考相机的绝对位姿
            for (size_t snapshot_idx = 0; snapshot_idx < camera_rig.NumSnapshots();
                 ++snapshot_idx) {
                if (!reconstruction.IsImageRegistered(camera_rig.Snapshots()[snapshot_idx][0])) {
                    continue;// 如果快照中的第一个图像未注册，则跳过该快照
                }
                //std::cout << "\n\nsnapshot_idx:" << snapshot_idx << "\n";
                camera_rig.ComputeAbsolutePose(snapshot_idx, reconstruction,
                                               &rig_qvecs[snapshot_idx],
                                               &rig_tvecs[snapshot_idx]);
                // 记录图片id对应的snapshot的相机组的位姿
                for (const auto image_id: camera_rig.Snapshots()[snapshot_idx]) {
                    /*std::cout << "\nimage_id:" << image_id << " ";
                    for (int i = 0; i < 4; i++) {
                        std::cout << rig_qvecs[snapshot_idx][i] << " ";
                    }*/
                    image_id_to_rig_qvec_.emplace(image_id, &rig_qvecs[snapshot_idx]);
                    image_id_to_rig_tvec_.emplace(image_id, &rig_tvecs[snapshot_idx]);
                }
            }
        }
    }

    // 对于ceres需要采用流型的方法处理四元数
    // 将所有要用于优化的四元数（图片本身的和相机组的）
    // 选择合适的参数化方式以进行计算
    void RigBundleAdjuster::ParameterizeCameraRigs(Reconstruction *reconstruction) {
        for (double *qvec_data: parameterized_qvec_data_) {
            /*把qvec_data中每个元素打印出来
            std::cout << "qvec_data:" << std::endl;
            for (int i = 0; i < 4; ++i) {
                std::cout << qvec_data[i] << " ";
            }*/
            SetQuaternionManifold(problem_.get(), qvec_data);
        }
    }

    void PrintSolverSummary(const ceres::Solver::Summary &summary) {
        std::cout << std::right << std::setw(16) << "Residuals : ";
        std::cout << std::left << summary.num_residuals_reduced << std::endl;

        std::cout << std::right << std::setw(16) << "Parameters : ";
        std::cout << std::left << summary.num_effective_parameters_reduced
                  << std::endl;

        std::cout << std::right << std::setw(16) << "Iterations : ";
        std::cout << std::left
                  << summary.num_successful_steps + summary.num_unsuccessful_steps
                  << std::endl;

        std::cout << std::right << std::setw(16) << "Time : ";
        std::cout << std::left << summary.total_time_in_seconds << " [s]"
                  << std::endl;

        std::cout << std::right << std::setw(16) << "Initial cost : ";
        std::cout << std::right << std::setprecision(6)
                  << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
                  << " [px]" << std::endl;

        std::cout << std::right << std::setw(16) << "Final cost : ";
        std::cout << std::right << std::setprecision(6)
                  << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
                  << " [px]" << std::endl;

        std::cout << std::right << std::setw(16) << "Termination : ";

        std::string termination = "";

        switch (summary.termination_type) {
            case ceres::CONVERGENCE:
                termination = "Convergence";
                break;
            case ceres::NO_CONVERGENCE:
                termination = "No convergence";
                break;
            case ceres::FAILURE:
                termination = "Failure";
                break;
            case ceres::USER_SUCCESS:
                termination = "User success";
                break;
            case ceres::USER_FAILURE:
                termination = "User failure";
                break;
            default:
                termination = "Unknown";
                break;
        }

        std::cout << std::right << termination << std::endl;
        std::cout << std::endl;
    }

}// namespace colmap
