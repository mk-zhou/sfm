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

#include "controllers/incremental_mapper.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#ifdef ENABLE_POSITION_PRIOR
#include "base/similarity_transform.h"// to estimate sim3
#include "util/eigen_alias_definition.h"
#include "util/numeric.h"
#endif
namespace colmap {
    namespace {

        size_t TriangulateImage(const IncrementalMapperOptions &options,
                                const Image &image, IncrementalMapper *mapper) {
            //std::cout << "  => Continued observations: " << image.NumPoints3D()
            //<< std::endl;
            const size_t num_tris =
                    mapper->TriangulateImage(options.Triangulation(), image.ImageId());
            //std::cout << "  => Added observations: " << num_tris << std::endl;
            return num_tris;
        }

        size_t TriangulateFrontWideImage(const IncrementalMapperOptions &options,
                                         const Image &image, IncrementalMapper *mapper) {
            //std::cout << "  => Continued observations: " << image.NumPoints3D()
            //<< std::endl;
            const size_t num_tris =
                    mapper->TriangulateFrontWideImage(options.Triangulation(), image.ImageId());
            //std::cout << "  => Added observations: " << num_tris << std::endl;
            return num_tris;
        }

        void AdjustGlobalBundle(const IncrementalMapperOptions &options,
                                IncrementalMapper *mapper) {
            BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();

            const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

            // Use stricter convergence criteria for first registered images.
            const size_t kMinNumRegImagesForFastBA = 10;
            if (num_reg_images < kMinNumRegImagesForFastBA) {
                custom_ba_options.solver_options.function_tolerance /= 10;
                custom_ba_options.solver_options.gradient_tolerance /= 10;
                custom_ba_options.solver_options.parameter_tolerance /= 10;
                custom_ba_options.solver_options.max_num_iterations *= 2;
                custom_ba_options.solver_options.max_linear_solver_iterations = 200;
            }

            PrintHeading1("Global bundle adjustment");
            if (options.ba_global_use_pba && !options.fix_existing_images &&
                num_reg_images >= kMinNumRegImagesForFastBA &&
                ParallelBundleAdjuster::IsSupported(custom_ba_options,
                                                    mapper->GetReconstruction())) {
                mapper->AdjustParallelGlobalBundle(
                        custom_ba_options, options.ParallelGlobalBundleAdjustment());
            } else {
                mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options, options.b_usable_prior);
            }
        }

        void AdjustFrontWideGlobalBundle(const IncrementalMapperOptions &options,
                                         IncrementalMapper *mapper) {
            BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();

            const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

            // Use stricter convergence criteria for first registered images.
            const size_t kMinNumRegImagesForFastBA = 10;
            if (num_reg_images < kMinNumRegImagesForFastBA) {
                custom_ba_options.solver_options.function_tolerance /= 10;
                custom_ba_options.solver_options.gradient_tolerance /= 10;
                custom_ba_options.solver_options.parameter_tolerance /= 10;
                custom_ba_options.solver_options.max_num_iterations *= 2;
                custom_ba_options.solver_options.max_linear_solver_iterations = 200;
            }

            PrintHeading1("Global bundle adjustment");
            if (options.ba_global_use_pba && !options.fix_existing_images &&
                num_reg_images >= kMinNumRegImagesForFastBA &&
                ParallelBundleAdjuster::IsSupported(custom_ba_options,
                                                    mapper->GetReconstruction())) {
                mapper->AdjustParallelGlobalBundle(
                        custom_ba_options, options.ParallelGlobalBundleAdjustment());
            } else {
                mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options, options.b_usable_prior);
            }
        }

        void RigAdjustGlobalBundle(const IncrementalMapperOptions &options,
                                   IncrementalMapper *mapper, std::vector<CameraRig> *camera_rigs) {
            BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();

            const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

            // Use stricter convergence criteria for first registered images.
            const size_t kMinNumRegImagesForFastBA = 10;
            if (num_reg_images < kMinNumRegImagesForFastBA) {
                custom_ba_options.solver_options.function_tolerance /= 10;
                custom_ba_options.solver_options.gradient_tolerance /= 10;
                custom_ba_options.solver_options.parameter_tolerance /= 10;
                custom_ba_options.solver_options.max_num_iterations *= 2;
                custom_ba_options.solver_options.max_linear_solver_iterations = 200;
            }

            PrintHeading1("Global bundle adjustment");
            if (options.ba_global_use_pba && !options.fix_existing_images &&
                num_reg_images >= kMinNumRegImagesForFastBA &&
                ParallelBundleAdjuster::IsSupported(custom_ba_options,
                                                    mapper->GetReconstruction())) {
                mapper->AdjustParallelGlobalBundle(
                        custom_ba_options, options.ParallelGlobalBundleAdjustment());
            } else {
                mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options, options.b_usable_prior);
            }
        }

        void IterativeLocalRefinement(const IncrementalMapperOptions &options,
                                      const image_t image_id,
                                      IncrementalMapper *mapper) {
            auto ba_options = options.LocalBundleAdjustment();
            for (int i = 0; i < options.ba_local_max_refinements; ++i) {
                const auto report = mapper->AdjustLocalBundle(
                        options.Mapper(), ba_options, options.Triangulation(), image_id,
                        mapper->GetModifiedPoints3D());
                std::cout << "  => Merged observations: " << report.num_merged_observations << std::endl;
                std::cout << "  => Completed observations: " << report.num_completed_observations << std::endl;
                std::cout << "  => Filtered observations: " << report.num_filtered_observations << std::endl;
                const double changed = report.num_adjusted_observations == 0
                                               ? 0
                                               : (report.num_merged_observations +
                                                  report.num_completed_observations +
                                                  report.num_filtered_observations) /
                                                         static_cast<double>(report.num_adjusted_observations);

                if (changed < options.ba_local_max_refinement_change) {
                    break;
                }
                // Only use robust cost function for first iteration.
                ba_options.loss_function_type =
                        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
            }
            mapper->ClearModifiedPoints3D();
        }

        void IterativeFrontWideLocalRefinement(const IncrementalMapperOptions &options,
                                               const image_t image_id,
                                               IncrementalMapper *mapper) {
            auto ba_options = options.LocalBundleAdjustment();
            for (int i = 0; i < options.ba_local_max_refinements; ++i) {
                const auto report = mapper->AdjustFrontWideLocalBundle(
                        options.Mapper(), ba_options, options.Triangulation(), image_id,
                        mapper->GetModifiedPoints3D());
                std::cout << "  => Merged observations: " << report.num_merged_observations
                          << std::endl;
                std::cout << "  => Completed observations: "
                          << report.num_completed_observations << std::endl;
                std::cout << "  => Filtered observations: "
                          << report.num_filtered_observations << std::endl;
                const double changed =
                        report.num_adjusted_observations == 0
                                ? 0
                                : (report.num_merged_observations +
                                   report.num_completed_observations +
                                   report.num_filtered_observations) /
                                          static_cast<double>(report.num_adjusted_observations);
                std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                          << std::endl;
                if (changed < options.ba_local_max_refinement_change) {
                    break;
                }
                // Only use robust cost function for first iteration.
                ba_options.loss_function_type =
                        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
            }
            mapper->ClearModifiedPoints3D();
        }

        void IterativeGlobalRefinement(const IncrementalMapperOptions &options,
                                       IncrementalMapper *mapper) {
            PrintHeading1("Retriangulation");
            CompleteAndMergeTracks(options, mapper);
            std::cout << "  => Retriangulated observations: "
                      << mapper->Retriangulate(options.Triangulation()) << std::endl;

            for (int i = 0; i < options.ba_global_max_refinements; ++i) {
                const size_t num_observations =
                        mapper->GetReconstruction().ComputeNumObservations();
                size_t num_changed_observations = 0;
                AdjustGlobalBundle(options, mapper);
                num_changed_observations += CompleteAndMergeTracks(options, mapper);
                num_changed_observations += FilterPoints(options, mapper);
                const double changed =
                        num_observations == 0
                                ? 0
                                : static_cast<double>(num_changed_observations) / num_observations;
                std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                          << std::endl;
                if (changed < options.ba_global_max_refinement_change) {
                    break;
                }
            }

            FilterImages(options, mapper);
        }

        void IterativeFrontWideGlobalRefinement(const IncrementalMapperOptions &options,
                                                IncrementalMapper *mapper) {
            PrintHeading1("Retriangulation");
            CompleteAndMergeTracks(options, mapper);
            std::cout << "  => Retriangulated observations: "
                      << mapper->Retriangulate(options.Triangulation()) << std::endl;

            for (int i = 0; i < options.ba_global_max_refinements; ++i) {
                const size_t num_observations =
                        mapper->GetReconstruction().ComputeNumObservations();
                size_t num_changed_observations = 0;
                AdjustGlobalBundle(options, mapper);
                num_changed_observations += CompleteAndMergeTracks(options, mapper);
                num_changed_observations += FilterFrontWidePoints(options, mapper);
                const double changed =
                        num_observations == 0
                                ? 0
                                : static_cast<double>(num_changed_observations) / num_observations;
                std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                          << std::endl;
                if (changed < options.ba_global_max_refinement_change) {
                    break;
                }
            }

            FilterImages(options, mapper);
        }

        void IterativeRigGlobalRefinement(const IncrementalMapperOptions &options, IncrementalMapper *mapper, std::vector<CameraRig> *camera_rigs, std::string output_path) {
            PrintHeading1("Retriangulation");
            CompleteAndMergeTracks(options, mapper);
            auto filter = mapper->MyGetReconstruction().FilterObservationsWithNegativeDepth();
            std::cout << "  => Retriangulated observations: "
                      << mapper->Retriangulate(options.Triangulation()) << std::endl;

            for (int i = 0; i < options.ba_global_max_refinements - 2; ++i) {
                const std::vector<image_t> &reg_image_ids = mapper->GetReconstruction().RegImageIds();

#ifdef ENABLE_POSITION_PRIOR
                //std::cout << "running incremental rig mapper based on enable position prior" << std::endl;
                bool b_usable_prior                     = true;
                double pose_center_robust_fitting_error = 0.0;
                // - Estimate SIM3 transformation between SFM position and prior position
                // - Compute a robust X-Y affine transformation & apply it
                // - This early transformation enhance the conditionning (solution closer to the Prior coordinate system)
                // 计算每个reg image目前的光心坐标和先验给出的光心坐标
                if (reg_image_ids.size() >= 3 && b_usable_prior) {
                    //定义一个sim3变换
                    SimilarityTransform3 sim3(1, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d(0, 0, 0));
                    //得到每个reg image的光心坐标和先验的光心坐标
                    std::vector<Eigen::Vector3d> X_SfM, X_GPS;
                    for (const image_t image_id: reg_image_ids) {
                        const Image &image = mapper->GetReconstruction().Image(image_id);
                        //std::cout<<image.QvecPrior()[0]<<image.QvecPrior()[1]<<image.QvecPrior()[2]<<image.QvecPrior()[3]<<std::endl;
                        if (!image.HasTvecPrior() || !image.HasTvecPrior()) {
                            std::cout << "running incremental mapper with no prior" << std::endl;
                            continue;
                        }
                        Eigen::Vector3d sfm_pose_center = image.ProjectionCenter();
                        Eigen::Vector3d gps_pose_center = ProjectionCenterFromPose(image.QvecPrior(), image.TvecPrior());
                        //std::cout<<image.QvecPrior()[0]<<image.QvecPrior()[1]<<image.QvecPrior()[2]<<image.QvecPrior()[3]<<std::endl;
                        X_SfM.push_back(sfm_pose_center);
                        X_GPS.push_back(gps_pose_center);
                    }
                    bool use_robust_estimation = true;// whether to use RANSAC to estimate sim3
                    int min_inlier_positions   = 3;
                    if (use_robust_estimation) {
                        // Only compute the alignment if there are enough correspondences.
                        if (X_SfM.size() < static_cast<size_t>(min_inlier_positions)) {
                            std::cout << "no enough correspondences" << std::endl;
                            return;
                        }
                        // Robustly estimate transformation using RANSAC.
                        // 利用ransac估计sfm坐标系到世界坐标系的变换
                        RANSACOptions ransac_options;
                        ransac_options.max_error = 10;
                        //Locally Optimized RANSAC
                        LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>> ransac(ransac_options);
                        //使用ransac基于一组对应点做sim3变换，并生成report
                        //report是ransac的一个结构体
                        const auto report = ransac.Estimate(X_SfM, X_GPS);
                        if (report.support.num_inliers < static_cast<size_t>(min_inlier_positions)) {
                            bool success = sim3.Estimate(X_SfM, X_GPS);
                        } else
                            sim3 = SimilarityTransform3(report.model);
                    } else {
                        // use all positions
                        bool success = sim3.Estimate(X_SfM, X_GPS);
                    }
                    mapper->MyGetReconstruction().Transform(sim3);// 对重建对象进行相机做sim3变换
                    {
                        std::vector<Eigen::Vector3d> X_SfM_new;// 存储转换后的姿态
                        for (const image_t image_id: reg_image_ids) {
                            const Image &image = mapper->GetReconstruction().Image(image_id);
                            if (!image.HasTvecPrior() || !image.HasTvecPrior())
                                continue;
                            Eigen::Vector3d sfm_pose_center = image.ProjectionCenter();// 获取图像的投影中心（姿态）
                            X_SfM_new.push_back(sfm_pose_center);                      // 将投影中心添加到 X_SfM_new 向量中
                        }
                        // 计算重建点云特征点与 GPS 点的残差
                        Eigen::VectorXd residual = (Eigen::Map<Mat3X>(X_SfM_new[0].data(), 3, X_SfM_new.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
                        // 对残差向量进行排序
                        std::sort(residual.data(), residual.data() + residual.size());
                        // 获取鲁棒拟合误差（使用中位数）
                        pose_center_robust_fitting_error = residual(residual.size() / 2);
                    }
                } else {
                    /* at least 3 registed images */
                    // nothing to do
                }
#endif
                bool estimate_rig_relative_poses = true;  // 是否估计相机组的相对位姿
                RigBundleAdjuster::Options rig_ba_options;// 相机组Bundle Adjustment选项
                OptionManager my_options;                 // 选项管理器
                BundleAdjustmentConfig config;            // Bundle Adjustment配置
                                                          // 将所有已注册的图像添加到Bundle Adjustment配置中
                                                          // Fix the existing images, if option specified.

                for (const auto image_id: mapper->GetReconstruction().RegImageIds()) {
                    //std::cout << "image_id " << image_id << std::endl;
                    config.AddImage(image_id);
                }
                if (!mapper->existing_image_ids_.empty()) {
                    std::cout << "rig fix_existing_images" << std::endl;
                    for (const image_t image_id: reg_image_ids) {
                        if (mapper->existing_image_ids_.count(image_id)) {
                            //std::cout << "image_id" << image_id << std::endl;
                            config.SetConstantPose(image_id);
                        }
                    }
                }
#ifdef ENABLE_POSITION_PRIOR
                config.SetFittingError(pose_center_robust_fitting_error);
                // TODO: need to automatically adjust XYZ weight
                const Eigen::Vector3d xyz_weight = Eigen::Vector3d::Constant(1.0);
                config.SetPriorPoseWeight(xyz_weight);
#endif

                PrintHeading1("Rig bundle adjustment");                                                // 打印标题：相机组Bundle Adjustment
                BundleAdjustmentOptions ba_options                     = *my_options.bundle_adjustment;// 设置Bundle Adjustment选项
                ba_options.solver_options.minimizer_progress_to_stdout = false;                        // 不将求解器的进度输出到控制台
                RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);                 // 创建相机组Bundle Adjuster对象
                //std::cout<<"test mapper->MyGetReconstruction()  "<<mapper->MyGetReconstruction()->NumCameras();
                CHECK(bundle_adjuster.Solve(&(mapper->MyGetReconstruction()), camera_rigs));// 执行相机组Bundle Adjustment
#ifdef ENABLE_POSITION_PRIOR
                if (b_usable_prior) {
                    std::vector<Eigen::Vector3d> X_SfM, X_GPS;
                    for (const image_t image_id: reg_image_ids) {
                        const Image &image = mapper->MyGetReconstruction().Image(image_id);
                        if (!image.HasTvecPrior() || !image.HasTvecPrior())
                            continue;
                        Eigen::Vector3d sfm_pose_center = image.ProjectionCenter();
                        Eigen::Vector3d gps_pose_center = ProjectionCenterFromPose(image.QvecPrior(), image.TvecPrior());
                        X_SfM.push_back(sfm_pose_center);
                        X_GPS.push_back(gps_pose_center);
                    }
                    // Compute the registration fitting error (once BA with Prior have been used):
                    if (X_GPS.size() > 3) {
                        // Compute the median residual error
                        Eigen::VectorXd residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
                        std::cout
                                << "Pose prior statistics (user units):\n"
                                << " - Starting median fitting error: " << pose_center_robust_fitting_error << "\n"
                                << " - Final fitting error:";
                        minMaxMeanMedian<Eigen::VectorXd::Scalar>(residual.data(), residual.data() + residual.size());
                    }
                }
#else
                        // Normalize scene for numerical stability and
                // to avoid large scale changes in viewer.
                reconstruction_->Normalize();
#endif
                FilterPoints(options, mapper);
                FilterImages(options, mapper);
                if (!output_path.empty() && i == options.ba_global_max_refinements - 3) {
                    std::cout << "test output_path " << output_path << std::endl;
                    bundle_adjuster.WriteCarPoseAndExtri(&mapper->MyGetReconstruction(), camera_rigs, output_path);
                }
            }
        }

        void IterativeRigLocalRefinement(const IncrementalMapperOptions &options, IncrementalMapper *mapper, const image_t ref_image_id, std::vector<CameraRig> *camera_rigs) {

            for (int i = 0; i < options.ba_local_max_refinements; ++i) {
                const std::vector<image_t> &reg_image_ids = mapper->GetReconstruction().RegImageIds();
                bool estimate_rig_relative_poses          = true;// 是否估计相机组的相对位姿
                RigBundleAdjuster::Options rig_ba_options;       // 相机组Bundle Adjustment选项
                OptionManager my_options;                        // 选项管理器
                BundleAdjustmentConfig config;                   // Bundle Adjustment配置
                // 将所有已注册的图像添加到Bundle Adjustment配置中
                std::vector<image_t> local_ref_image_ids;
                std::cout << "mapper->GetReconstruction().NumCameras()=" << mapper->GetReconstruction().NumCameras();

                for (image_t n = 1; n < 10; n++) {
                    if (local_ref_image_ids.size() >= 10) {
                        break;// 如果已经达到13个元素，跳出循环
                    }
                    if (mapper->GetReconstruction().ExistsImage(ref_image_id + n)) {
                        if (mapper->GetReconstruction().IsImageRegistered(ref_image_id + n) &&
                            mapper->GetReconstruction().Image(ref_image_id + n).CameraId() == mapper->GetReconstruction().NumCameras()) {
                            local_ref_image_ids.push_back(ref_image_id + n);
                        }
                    }

                    if (local_ref_image_ids.size() >= 10) {
                        break;// 如果已经达到10个元素，跳出循环
                    }
                    if (mapper->GetReconstruction().ExistsImage(ref_image_id - n)) {
                        if (mapper->GetReconstruction().IsImageRegistered(ref_image_id - n) &&
                            mapper->GetReconstruction().Image(ref_image_id - n).CameraId() == mapper->GetReconstruction().NumCameras()) {
                            //std::cout << "adding image_id=" << reg_image_id - i << std::endl;
                            local_ref_image_ids.push_back(ref_image_id - n);
                        }
                    }
                }
                CameraRig &first_camera_rig = camera_rigs->at(0);
                for (const auto image_id: local_ref_image_ids) {
                    std::vector<image_t> image_in_rig_ids = first_camera_rig.GetImageId(first_camera_rig.GetSnapshotId(image_id));
                    for (const auto image_in_rig_id: image_in_rig_ids) {
                        config.AddImage(image_in_rig_id);
                    }
                }
                PrintHeading1("Local Rig bundle adjustment");                                          // 打印标题：相机组Bundle Adjustment
                BundleAdjustmentOptions ba_options                     = *my_options.bundle_adjustment;// 设置Bundle Adjustment选项
                ba_options.solver_options.minimizer_progress_to_stdout = false;                        // 不将求解器的进度输出到控制台
                RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);                 // 创建相机组Bundle Adjuster对象
                //std::cout<<"test mapper->MyGetReconstruction()  "<<mapper->MyGetReconstruction()->NumCameras();
                CHECK(bundle_adjuster.Solve(&(mapper->MyGetReconstruction()), camera_rigs));// 执行相机组Bundle Adjustment
                FilterPoints(options, mapper);
                FilterImages(options, mapper);
            }
        }
        void ExtractColors(const std::string &image_path, const image_t image_id,
                           Reconstruction *reconstruction) {
            if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
                std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                                          reconstruction->Image(image_id).Name().c_str(),
                                          image_path.c_str())
                          << std::endl;
            }
        }

        void WriteSnapshot(const Reconstruction &reconstruction,
                           const std::string &snapshot_path) {
            PrintHeading1("Creating snapshot");
            // Get the current timestamp in milliseconds.
            const size_t timestamp =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch())
                            .count();
            // Write reconstruction to unique path with current timestamp.
            const std::string path =
                    JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
            CreateDirIfNotExists(path);
            std::cout << "  => Writing to " << path << std::endl;
            reconstruction.Write(path);
        }

    }// namespace

    size_t FilterPoints(const IncrementalMapperOptions &options,
                        IncrementalMapper *mapper) {
        const size_t num_filtered_observations =
                mapper->FilterPoints(options.Mapper());
        std::cout << "  => Filtered observations: " << num_filtered_observations
                  << std::endl;
        return num_filtered_observations;
    }

    size_t FilterFrontWidePoints(const IncrementalMapperOptions &options,
                                 IncrementalMapper *mapper) {
        const size_t num_filtered_observations =
                mapper->FilterFrontWidePoints(options.Mapper());
        std::cout << "  => Filtered observations: " << num_filtered_observations
                  << std::endl;
        return num_filtered_observations;
    }

    size_t FilterImages(const IncrementalMapperOptions &options,
                        IncrementalMapper *mapper) {
        const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
        std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
        return num_filtered_images;
    }

    size_t CompleteAndMergeTracks(const IncrementalMapperOptions &options,
                                  IncrementalMapper *mapper) {
        const size_t num_completed_observations =
                mapper->CompleteTracks(options.Triangulation());
        std::cout << "  => Completed observations: " << num_completed_observations
                  << std::endl;
        const size_t num_merged_observations =
                mapper->MergeTracks(options.Triangulation());
        std::cout << "  => Merged observations: " << num_merged_observations
                  << std::endl;
        return num_completed_observations + num_merged_observations;
    }

    IncrementalMapper::Options IncrementalMapperOptions::Mapper() const {
        IncrementalMapper::Options options   = mapper;
        options.abs_pose_refine_focal_length = ba_refine_focal_length;
        options.abs_pose_refine_extra_params = ba_refine_extra_params;
        options.min_focal_length_ratio       = min_focal_length_ratio;
        options.max_focal_length_ratio       = max_focal_length_ratio;
        options.max_extra_param              = max_extra_param;
        options.num_threads                  = num_threads;
        options.local_ba_num_images          = ba_local_num_images;
        options.fix_existing_images          = fix_existing_images;
        return options;
    }

    IncrementalTriangulator::Options IncrementalMapperOptions::Triangulation()
            const {
        IncrementalTriangulator::Options options = triangulation;
        options.min_focal_length_ratio           = min_focal_length_ratio;
        options.max_focal_length_ratio           = max_focal_length_ratio;
        options.max_extra_param                  = max_extra_param;
        return options;
    }

    BundleAdjustmentOptions IncrementalMapperOptions::LocalBundleAdjustment()
            const {
        BundleAdjustmentOptions options;
        options.solver_options.function_tolerance           = ba_local_function_tolerance;
        options.solver_options.gradient_tolerance           = 10.0;
        options.solver_options.parameter_tolerance          = 0.0;
        options.solver_options.max_num_iterations           = ba_local_max_num_iterations;
        options.solver_options.max_linear_solver_iterations = 100;
        options.solver_options.minimizer_progress_to_stdout = false;
        options.solver_options.num_threads                  = num_threads;
#if CERES_VERSION_MAJOR < 2
        options.solver_options.num_linear_solver_threads = num_threads;
#endif// CERES_VERSION_MAJOR
        options.print_summary          = true;
        options.refine_focal_length    = ba_refine_focal_length;
        options.refine_principal_point = ba_refine_principal_point;
        options.refine_extra_params    = ba_refine_extra_params;
        options.min_num_residuals_for_multi_threading =
                ba_min_num_residuals_for_multi_threading;
        options.loss_function_scale = 1.0;
        options.loss_function_type =
                BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
        return options;
    }

    BundleAdjustmentOptions IncrementalMapperOptions::GlobalBundleAdjustment()
            const {
        BundleAdjustmentOptions options;
        options.solver_options.function_tolerance           = ba_global_function_tolerance;
        options.solver_options.gradient_tolerance           = 1.0;
        options.solver_options.parameter_tolerance          = 0.0;
        options.solver_options.max_num_iterations           = ba_global_max_num_iterations;
        options.solver_options.max_linear_solver_iterations = 100;
        options.solver_options.minimizer_progress_to_stdout = false;
        options.solver_options.num_threads                  = num_threads;
#if CERES_VERSION_MAJOR < 2
        options.solver_options.num_linear_solver_threads = num_threads;
#endif// CERES_VERSION_MAJOR
        options.print_summary          = true;
        options.refine_focal_length    = ba_refine_focal_length;
        options.refine_principal_point = ba_refine_principal_point;
        options.refine_extra_params    = ba_refine_extra_params;
        options.min_num_residuals_for_multi_threading =
                ba_min_num_residuals_for_multi_threading;
        options.loss_function_type =
                BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
        return options;
    }

    ParallelBundleAdjuster::Options IncrementalMapperOptions::ParallelGlobalBundleAdjustment() const {
        ParallelBundleAdjuster::Options options;
        options.max_num_iterations = ba_global_max_num_iterations;
        options.print_summary      = true;
        options.gpu_index          = ba_global_pba_gpu_index;
        options.num_threads        = num_threads;
        options.min_num_residuals_for_multi_threading =
                ba_min_num_residuals_for_multi_threading;
        return options;
    }

    bool IncrementalMapperOptions::Check() const {
        CHECK_OPTION_GT(min_num_matches, 0);
        CHECK_OPTION_GT(max_num_models, 0);
        CHECK_OPTION_GT(max_model_overlap, 0);
        CHECK_OPTION_GE(min_model_size, 0);
        CHECK_OPTION_GT(init_num_trials, 0);
        CHECK_OPTION_GT(min_focal_length_ratio, 0);
        CHECK_OPTION_GT(max_focal_length_ratio, 0);
        CHECK_OPTION_GE(max_extra_param, 0);
        CHECK_OPTION_GE(ba_local_num_images, 2);
        CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
        CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
        CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
        CHECK_OPTION_GT(ba_global_images_freq, 0);
        CHECK_OPTION_GT(ba_global_points_freq, 0);
        CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
        CHECK_OPTION_GT(ba_local_max_refinements, 0);
        CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
        CHECK_OPTION_GT(ba_global_max_refinements, 0);
        CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
        CHECK_OPTION_GE(snapshot_images_freq, 0);
        CHECK_OPTION(Mapper().Check());
        CHECK_OPTION(Triangulation().Check());
        return true;
    }

    IncrementalMapperController::IncrementalMapperController(
            const IncrementalMapperOptions *options, const std::string &image_path,
            const std::string &database_path,
            ReconstructionManager *reconstruction_manager)
        : options_(options),
          image_path_(image_path),
          database_path_(database_path),
          reconstruction_manager_(reconstruction_manager) {
        CHECK(options_->Check());
        RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
        RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
        RegisterCallback(LAST_IMAGE_REG_CALLBACK);
    }

    IncrementalMapperController::IncrementalMapperController(
            const IncrementalMapperOptions *options, std::string image_path,
            std::string database_path, std::string rig_config_path,
            std::string ref_image_list_path, std::string output_path,
            ReconstructionManager *reconstruction_manager)
        : options_(options),
          image_path_(std::move(image_path)),
          database_path_(std::move(database_path)),
          rig_config_path_(std::move(rig_config_path)),
          ref_image_list_path_(std::move(ref_image_list_path)),
          output_path_(std::move(output_path)),
          reconstruction_manager_(reconstruction_manager) {
        CHECK(options_->Check());
        RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
        RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
        RegisterCallback(LAST_IMAGE_REG_CALLBACK);
    }

    void IncrementalMapperController::Run() {
        //加载数据库的函数
        if (!LoadDatabase()) {
            return;
        }

        IncrementalMapper::Options init_mapper_options = options_->Mapper();
        if (rig_config_path_ != "" and ref_image_list_path_ == "") {
            RigReconstruct(init_mapper_options);
        } else if (rig_config_path_ != "" and ref_image_list_path_ != "") {
            MergeReconstruct(init_mapper_options);
        } else {
            Reconstruct(init_mapper_options);
        }
        const size_t kNumInitRelaxations = 2;
        //是否找到初始对，如果没找到放宽条件再找，最多找kNumInitRelaxations次
        for (size_t i = 0; i < kNumInitRelaxations; ++i) {
            if (reconstruction_manager_->Size() > 0 || IsStopped()) {
                std::cout << "stopped" << std::endl;
                break;
            }
            //没找到，放宽条件重新找
            std::cout << "  => Relaxing the initialization constraints." << std::endl;
            init_mapper_options.init_min_num_inliers /= 2;
            Reconstruct(init_mapper_options);

            if (reconstruction_manager_->Size() > 0 || IsStopped()) {
                break;
            }
            ////没找到，，放宽条件重新找
            std::cout << "  => Relaxing the initialization constraints." << std::endl;
            init_mapper_options.init_min_tri_angle /= 2;
            Reconstruct(init_mapper_options);
        }

        std::cout << std::endl;
        GetTimer().PrintMinutes();
    }

    bool IncrementalMapperController::LoadDatabase() {
        PrintHeading1("Loading database");

        // Make sure images of the given reconstruction are also included when
        // manually specifying images for the reconstrunstruction procedure.
        std::unordered_set<std::string> image_names = options_->image_names;
        if (reconstruction_manager_->Size() == 1 && !options_->image_names.empty()) {
            const Reconstruction &reconstruction = reconstruction_manager_->Get(0);
            for (const image_t image_id: reconstruction.RegImageIds()) {
                const auto &image = reconstruction.Image(image_id);
                image_names.insert(image.Name());
            }
        }

        Database database(database_path_);
        Timer timer;
        timer.Start();
        const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
        //通过database_cache_储存数据库的内容
        database_cache_.Load(database, min_num_matches, options_->ignore_watermarks,
                             image_names);
        std::cout << std::endl;
        timer.PrintMinutes();

        std::cout << std::endl;

        if (database_cache_.NumImages() == 0) {
            std::cout << "WARNING: No images with matches found in the database."
                      << std::endl
                      << std::endl;
            return false;
        }

        return true;
    }

    std::vector<CameraRig> IncrementalMapperController::ReadCameraRigConfigFromImageName(const std::string &rig_config_path,
                                                                                         const std::string &databae_path) {
        // 从rig_config_path读取json文件
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(rig_config_path.c_str(), pt);

        // 初始化camera_rigs
        std::vector<CameraRig> camera_rigs;
        // 遍历rig_config中的每一个rig_config
        for (const auto &rig_config: pt) {
            CameraRig camera_rig;

            // 将每一个camera的image_prefixes放入image_prefixes中
            std::vector<std::string> image_prefixes;
            for (const auto &camera: rig_config.second.get_child("cameras")) {
                const int camera_id = camera.second.get<int>("camera_id");
                //得到所有的前缀
                image_prefixes.push_back(camera.second.get<std::string>("image_prefix"));
                Eigen::Vector3d rel_tvec;
                Eigen::Vector4d rel_qvec;
                int index          = 0;
                auto rel_tvec_node = camera.second.get_child_optional("rel_tvec");
                //如果rel_tvec和rel_qvec都有的话就采用config里面的相对位姿
                //如果其中一个没有的话，就通过计算平均来得到相对位姿
                if (rel_tvec_node) {
                    for (const auto &node: rel_tvec_node.get()) {
                        rel_tvec[index++] = node.second.get_value<double>();
                    }
                } else {
                    //estimate_rig_relative_poses = true;
                }
                index              = 0;
                auto rel_qvec_node = camera.second.get_child_optional("rel_qvec");
                if (rel_qvec_node) {
                    for (const auto &node: rel_qvec_node.get()) {
                        rel_qvec[index++] = node.second.get_value<double>();
                    }
                }
                //else {
                //  estimate_rig_relative_poses = true;
                //}
                //当json文件里面没有相对位姿时才进行估计
                camera_rig.AddCamera(camera_id, rel_qvec, rel_tvec);
            }

            // 设置ref_camera_id
            camera_rig.SetRefCameraId(rig_config.second.get<int>("ref_camera_id"));

            //image_list = GetRecursiveFileList(image_path);
            // 将每一个snapshot放入snapshots中

            // 获取图像对象的 ID
            Database database(database_path_);
            std::vector<colmap::image_t> image_ids         = database.ReadAllImagesIds();
            std::vector<colmap::MySnapshot> snapshots_list = database.ReadAllMySnapshots();
            std::unordered_map<my_snapshot_t, std::vector<image_t>> snapshots;// 创建一个无序映射，用于存储快照的信息
            //const colmap::ImageID image_id = image.ImageId();
            // 遍历图像 ID 列表
            for (const auto &image_id: image_ids) {// 遍历数据库对象中的所有图像ID
                const auto &image = database.ReadImage(image_id);
                //std::cout << "image_id: " << image_id << std::endl;
                //std::cout << "image_name: " << image.Name() << std::endl;
                // 获取当前图像ID对应的图像对象
                for (const auto &image_prefix: image_prefixes) {
                    //std::cout << image_prefix << std::endl;
                    // 遍历图像前缀列表
                    if (image.Name().find(image_prefix) != std::string::npos) {    // 检查当前图像的名称是否包含当前图像前缀
                        const my_snapshot_t image_snapshot_id = image.SnapshotId();// 获取当前图像的快照ID
                        //std::cout << "image_snapshot_id: " << image_snapshot_id << std::endl;
                        snapshots[image_snapshot_id].push_back(image_id);// 将当前图像ID添加到对应图像后缀的快照中
                        // 如果该后缀的快照尚不存在，则先创建一个新的键，并将当前图像ID添加到对应的向量中
                    }
                }
            }
            // 遍历snapshots，检查是否有ref_camera
            for (const auto &snapshot: snapshots) {
                bool has_ref_camera = false;
                //std::cout << std::endl<< "snapshot.first  " << snapshot.first << std::endl;
                //std::cout << "image_id  ";
                for (const auto image_id: snapshot.second) {
                    //std::cout << image_id << "  " << std::endl;
                    const auto &image = database.ReadImage(image_id);
                    if (image.CameraId() == camera_rig.RefCameraId()) {
                        has_ref_camera = true;
                    }
                }

                if (has_ref_camera) {
                    //std::cout << "snapshot.second.size()  " << snapshot.second.size() << std::endl;
                    camera_rig.AddSnapshot(snapshot.second);
                }
            }

            // 检查camera_rig是否满足构建要求
            // camera_rig.Check(reconstruction);
            /*if (estimate_rig_relative_poses) {
      PrintHeading2("Estimating relative rig poses");
      if (!camera_rig.ComputeRelativePoses(reconstruction)) {
        std::cout << "WARN: Failed to estimate rig poses from reconstruction; "
                     "cannot use rig BA"
                  << std::endl;
        return std::vector<CameraRig>();
      }
    }*/

            camera_rigs.push_back(camera_rig);
        }

        return camera_rigs;
    }

    void IncrementalMapperController::MergeReconstruct(
            const IncrementalMapper::Options &init_mapper_options) {
        const bool kDiscardReconstruction = true;

        //////////////////////////////////////////////////////////////////////////////
        // Main loop
        //////////////////////////////////////////////////////////////////////////////

        IncrementalMapper mapper(&database_cache_);

        // Is there a sub-model before we start the reconstruction? I.e. the user
        // has imported an existing reconstruction.
        const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
        CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                        "single reconstruction, but "
                                                        "multiple are given.";

        for (int num_trials = 0; num_trials < options_->init_num_trials; ++num_trials) {
            BlockIfPaused();
            if (IsStopped()) {
                break;
            }

            size_t reconstruction_idx;
            if (!initial_reconstruction_given || num_trials > 0) {
                reconstruction_idx = reconstruction_manager_->Add();
            } else {
                reconstruction_idx = 0;
            }
            ////////////////////////////////////////////////////////////////////////////
            // 加载相机组信息
            ////////////////////////////////////////////////////////////////////////////
            Reconstruction &reconstruction =
                    reconstruction_manager_->Get(reconstruction_idx);
            PrintHeading1("Camera rig configuration");// 打印标题：相机组配置

            auto camera_rigs = ReadCameraRigConfigFromImageName(rig_config_path_, database_path_);// 读取相机组配置信息

            BundleAdjustmentConfig config;// Bundle Adjustment配置
            //std::cout << "reconstruction.NumImages()  " << reconstruction.NumImages() << std::endl;
            for (size_t i = 0; i < camera_rigs.size(); ++i) {// 遍历每个相机组
                const auto &camera_rig = camera_rigs[i];
                PrintHeading2(StringPrintf("Camera Rig %d", i + 1));                               // 打印标题：相机组编号
                std::cout << StringPrintf("Cameras: %d", camera_rig.NumCameras()) << std::endl;    // 打印相机数量
                std::cout << StringPrintf("Snapshots: %d", camera_rig.NumSnapshots()) << std::endl;// 打印快照数量
            }
            //初始化
            mapper.BeginReconstruction(&reconstruction);
            //////////////////////////////////////////////////////////////////////////////
            // 注册、三角化ref_image_list中的图像
            //////////////////////////////////////////////////////////////////////////////
            std::vector<image_t> ref_image_ids;
            for (auto ref_image_id: options_->ref_image_ids) {
                mapper.existing_image_ids_.insert(std::stoi(ref_image_id));
                const auto image_id   = std::stoi(ref_image_id);
                bool reg_next_success = mapper.RegisterNextImageInBeginning(options_->Mapper(), image_id);
            }
            for (image_t image_id: reconstruction.RegImageIds()) {
                Image &image = reconstruction.Image(image_id);
                TriangulateImage(*options_, image, &mapper);
                if (options_->extract_colors) {
                    ExtractColors(image_path_, image_id, &reconstruction);
                }
            }
            mapper.existing_image_ids_ = std::unordered_set<image_t>(
                    reconstruction.RegImageIds().begin(),
                    reconstruction.RegImageIds().end());
            //////////////////////////////////////////////////////////////////////////////
            // 注册、三角化、BA 需要计算的图像
            //////////////////////////////////////////////////////////////////////////////
            size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
            size_t ba_prev_num_reg_images       = reconstruction.NumRegImages();
            size_t ba_prev_num_points           = reconstruction.NumPoints3D();

            bool reg_next_success      = true;
            bool prev_reg_next_success = true;
            while (reg_next_success) {
                BlockIfPaused();
                if (IsStopped()) {
                    break;
                }

                reg_next_success = false;

                const std::vector<image_t> next_images =
                        mapper.FindNextImages(options_->Mapper());

                if (next_images.empty()) {
                    break;
                }

                for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
                    const image_t next_image_id = next_images[reg_trial];
                    const Image &next_image     = reconstruction.Image(next_image_id);

                    PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                               reconstruction.NumRegImages() + 1));

                    std::cout << StringPrintf("  => Image sees %d / %d points",
                                              next_image.NumVisiblePoints3D(),
                                              next_image.NumObservations())
                              << std::endl;

                    reg_next_success =
                            mapper.RegisterNextImage(options_->Mapper(), next_image_id);


                    if (reg_next_success) {
                        //std::cout<<"before triangulate, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                        TriangulateImage(*options_, next_image, &mapper);
                        //std::cout<<"after triangulate, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                        IterativeLocalRefinement(*options_, next_image_id, &mapper);
                        //std::cout<<"after local refinement, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                        if (reconstruction.NumRegImages() >=
                                    options_->ba_global_images_ratio * ba_prev_num_reg_images ||
                            reconstruction.NumRegImages() >=
                                    options_->ba_global_images_freq + ba_prev_num_reg_images ||
                            reconstruction.NumPoints3D() >=
                                    options_->ba_global_points_ratio * ba_prev_num_points ||
                            reconstruction.NumPoints3D() >=
                                    options_->ba_global_points_freq + ba_prev_num_points) {
                            //std::cout<<"before global ba, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                            IterativeGlobalRefinement(*options_, &mapper);
                            //std::cout<<"after global ba, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                            ba_prev_num_points     = reconstruction.NumPoints3D();
                            ba_prev_num_reg_images = reconstruction.NumRegImages();
                        }

                        if (options_->extract_colors) {
                            ExtractColors(image_path_, next_image_id, &reconstruction);
                        }

                        if (options_->snapshot_images_freq > 0 &&
                            reconstruction.NumRegImages() >=
                                    options_->snapshot_images_freq +
                                            snapshot_prev_num_reg_images) {
                            snapshot_prev_num_reg_images = reconstruction.NumRegImages();
                            WriteSnapshot(reconstruction, options_->snapshot_path);
                        }

                        Callback(NEXT_IMAGE_REG_CALLBACK);

                        break;
                    } else {
                        std::cout << "  => Could not register, trying another image."
                                  << std::endl;

                        // If initial pair fails to continue for some time,
                        // abort and try different initial pair.
                        const size_t kMinNumInitialRegTrials = 30;
                        if (reg_trial >= kMinNumInitialRegTrials &&
                            reconstruction.NumRegImages() <
                                    static_cast<size_t>(options_->min_model_size)) {
                            break;
                        }
                    }
                }
                //Callback(LAST_IMAGE_REG_CALLBACK);
                const size_t max_model_overlap =
                        static_cast<size_t>(options_->max_model_overlap);
                if (mapper.NumSharedRegImages() >= max_model_overlap) {
                    break;
                }

                // If no image could be registered, try a single final global iterative
                // bundle adjustment and try again to register one image. If this fails
                // once, then exit the incremental mapping.
                if (!reg_next_success && prev_reg_next_success) {
                    reg_next_success      = true;
                    prev_reg_next_success = false;
                    IterativeGlobalRefinement(*options_, &mapper);
                } else {
                    prev_reg_next_success = reg_next_success;
                }
            }

            if (IsStopped()) {
                const bool kDiscardReconstruction = false;
                mapper.EndReconstruction(kDiscardReconstruction);
                break;
            }

            // Only run final global BA, if last incremental BA was not global.
            if (reconstruction.NumRegImages() >= 2 &&
                reconstruction.NumRegImages() != ba_prev_num_reg_images &&
                reconstruction.NumPoints3D() != ba_prev_num_points) {
                IterativeGlobalRefinement(*options_, &mapper);
            }

            // If the total number of images is small then do not enforce the minimum
            // model size so that we can reconstruct small image collections.
            const size_t min_model_size =
                    std::min(database_cache_.NumImages(),
                             static_cast<size_t>(options_->min_model_size));
            IterativeRigGlobalRefinement(*options_, &mapper, &camera_rigs, "");
            if ((options_->multiple_models &&
                 reconstruction.NumRegImages() < min_model_size) ||
                reconstruction.NumRegImages() == 0) {
                mapper.EndReconstruction(kDiscardReconstruction);
                reconstruction_manager_->Delete(reconstruction_idx);
            } else {
                const bool kDiscardReconstruction = false;
                mapper.EndReconstruction(kDiscardReconstruction);
            }

            Callback(LAST_IMAGE_REG_CALLBACK);

            const size_t max_num_models = static_cast<size_t>(options_->max_num_models);
            if (initial_reconstruction_given || !options_->multiple_models ||
                reconstruction_manager_->Size() >= max_num_models ||
                mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {
                break;
            }
        }
    }
    void IncrementalMapperController::RigReconstruct(
            const IncrementalMapper::Options &init_mapper_options) {
        const bool kDiscardReconstruction = true;

        //////////////////////////////////////////////////////////////////////////////
        // Main loop
        //////////////////////////////////////////////////////////////////////////////
        std::string output_path = output_path_;
        IncrementalMapper mapper(&database_cache_);

        // Is there a sub-model before we start the reconstruction? I.e. the user
        // has imported an existing reconstruction.
        const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
        CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                        "single reconstruction, but "
                                                        "multiple are given.";

        for (int num_trials = 0; num_trials < options_->init_num_trials; ++num_trials) {
            BlockIfPaused();
            if (IsStopped()) {
                break;
            }

            size_t reconstruction_idx;
            if (!initial_reconstruction_given || num_trials > 0) {
                reconstruction_idx = reconstruction_manager_->Add();
            } else {
                reconstruction_idx = 0;
            }
            ////////////////////////////////////////////////////////////////////////////
            // 加载相机组信息
            ////////////////////////////////////////////////////////////////////////////
            Reconstruction &reconstruction =
                    reconstruction_manager_->Get(reconstruction_idx);
            PrintHeading1("Camera rig configuration");// 打印标题：相机组配置

            auto camera_rigs = ReadCameraRigConfigFromImageName(rig_config_path_, database_path_);// 读取相机组配置信息

            BundleAdjustmentConfig config;// Bundle Adjustment配置
            //std::cout << "reconstruction.NumImages()  " << reconstruction.NumImages() << std::endl;
            for (size_t i = 0; i < camera_rigs.size(); ++i) {// 遍历每个相机组
                const auto &camera_rig = camera_rigs[i];
                PrintHeading2(StringPrintf("Camera Rig %d", i + 1));                               // 打印标题：相机组编号
                std::cout << StringPrintf("Cameras: %d", camera_rig.NumCameras()) << std::endl;    // 打印相机数量
                std::cout << StringPrintf("Snapshots: %d", camera_rig.NumSnapshots()) << std::endl;// 打印快照数量
            }
            //初始化
            mapper.BeginReconstruction(&reconstruction);

            ////////////////////////////////////////////////////////////////////////////
            // Register initial pair
            ////////////////////////////////////////////////////////////////////////////
            {
                //std::cout << "reconstruction.NumImages()  " << reconstruction.NumImages() << std::endl;
                //找到则继续，找不到就跳出重建
                if (reconstruction.NumRegImages() == 0) {
                    image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
                    image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

                    // Try to find good initial pair.
                    // 遍历每一个图片组尝试
                    if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
                        PrintHeading1("Finding good initial image pair");
                        //跳转到 sfm/incremental_mapper.cc
                        //里面进行了两视几何，得到了两张图片的相对位姿关系
                        //这也是后面能做ba的原因
                        const bool find_init_success = mapper.FindInitialFrontWideImagePair(
                                init_mapper_options, &image_id1, &image_id2);

                        if (!find_init_success) {
                            std::cout << "  => No good initial image pair found." << std::endl;
                            mapper.EndReconstruction(kDiscardReconstruction);
                            reconstruction_manager_->Delete(reconstruction_idx);
                            break;
                        }
                    } else {
                        if (!reconstruction.ExistsImage(image_id1) ||
                            !reconstruction.ExistsImage(image_id2)) {
                            std::cout << StringPrintf(
                                                 "  => Initial image pair #%d and #%d do not exist.",
                                                 image_id1, image_id2)
                                      << std::endl;
                            mapper.EndReconstruction(kDiscardReconstruction);
                            reconstruction_manager_->Delete(reconstruction_idx);
                            return;
                        }
                    }

                    PrintHeading1(StringPrintf("Initializing with image pair #%d and #%d",
                                               image_id1, image_id2));
                    const bool reg_init_success = mapper.RegisterInitialFrontWideImagePair(
                            init_mapper_options, image_id1, image_id2);
                    //std::cout<<"reconstruction.NumPoints3D()  before BA"<<reconstruction.NumPoints3D();
                    if (!reg_init_success) {
                        std::cout << "  => Initialization failed - possible solutions:"
                                  << std::endl
                                  << "     - try to relax the initialization constraints"
                                  << std::endl
                                  << "     - manually select an initial image pair"
                                  << std::endl;
                        mapper.EndReconstruction(kDiscardReconstruction);
                        reconstruction_manager_->Delete(reconstruction_idx);
                        break;
                    }
                }

                Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
            }
            ////////////////////////////////////////////////////////////////////////////
            // Incremental mapping
            ////////////////////////////////////////////////////////////////////////////
            //记录最初两张图片重建的信息
            size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
            bool reg_next_success               = true;
            bool prev_reg_next_success          = true;
            std::vector<image_t> first_ten_snapshot_ids;
            int n = 0;
            std::vector<image_t> next_images;
            while (reg_next_success && reconstruction.NumRegImages() < 15 * reconstruction.NumCameras()) {
                BlockIfPaused();
                if (IsStopped()) {
                    break;
                }
                reg_next_success = false;
                if (reconstruction.NumRegImages() < 15) {
                    next_images = mapper.FindNextFrontWideImages(options_->Mapper());
                } else {
                    //清空next_images
                    next_images.clear();
                    for (size_t i = 0; i < reconstruction.NumRegImages(); ++i) {
                        auto image_id    = reconstruction.RegImageIds()[i];
                        auto snapshot_id = camera_rigs[0].GetSnapshotId(reconstruction.RegImageIds()[i]);
                        auto images_id   = camera_rigs[0].GetImageId(snapshot_id);

                        std::cout << "image_id=" << image_id << "  snapshot_id=" << snapshot_id << "  "
                                  << "all images_id: ";
                        for (auto id: images_id) {
                            std::cout << id << "   ";
                            // 检查 first_five_snapshot_ids 是否已经包含该 id
                            // 把所有没register的snapshot里面的图片加入待选队列
                            if (std::find(next_images.begin(), next_images.end(), id) == next_images.end() && reconstruction.ExistsImage(id) && !reconstruction.IsImageRegistered(id)) {
                                next_images.push_back(id);
                            }
                        }
                    }
                }

                if (next_images.empty()) {
                    break;
                }

                for (unsigned int next_image_id: next_images) {
                    const Image &next_image = reconstruction.Image(next_image_id);

                    PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                               reconstruction.NumRegImages() + 1));
                    //已经重建过的点
                    /*std::cout << StringPrintf("  => Image sees %d / %d points",
                                              next_image.NumVisiblePoints3D(),
                                              next_image.NumObservations())
                              << std::endl;*/

                    reg_next_success =
                            mapper.RegisterNextImageInBeginning(options_->Mapper(), next_image_id);
                    if (reg_next_success) {
                        Callback(NEXT_IMAGE_REG_CALLBACK);
                    }
                }
                for (image_t image_id: reconstruction.RegImageIds()) {
                    Image &image = reconstruction.Image(image_id);
                    TriangulateImage(*options_, image, &mapper);
                }
            }

            //Callback(LAST_IMAGE_REG_CALLBACK);
            reg_next_success = true;
            IterativeRigGlobalRefinement(*options_, &mapper, &camera_rigs, "");
            while (reg_next_success && reconstruction.NumRegImages() < reconstruction.NumImages()) {
                BlockIfPaused();
                if (IsStopped()) {
                    break;
                }
                reg_next_success = false;
                next_images      = mapper.FindNextImagesFromCameraRig(options_->Mapper(), camera_rigs[0]);
                for (auto id: next_images) { std::cout << id << " "; }
                if (next_images.empty()) {
                    break;
                }
                //把一个相机组里面所有的图片都配准、三角化、局部BA
                PrintHeading1(StringPrintf("Registering camera_rig (%d) ",
                                           reconstruction.NumRegImages() / reconstruction.NumCameras() + 1));

                reg_next_success =
                        mapper.RegisterNextImageRig(options_->Mapper(), &(camera_rigs[0]), next_images);
                for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
                    const image_t next_image_id = next_images[reg_trial];
                    const Image &next_image     = reconstruction.Image(next_image_id);
                    //已经重建过的点
                    //注册成功以后，做三角化
                    if (reg_next_success) {
                        TriangulateImage(*options_, next_image, &mapper);
                        //IterativeLocalRefinement(*options_, next_image_id, &mapper);
                        if (options_->extract_colors) {
                            ExtractColors(image_path_, next_image_id, &reconstruction);
                        }
                        if (options_->snapshot_images_freq > 0 && reconstruction.NumRegImages() >= options_->snapshot_images_freq + snapshot_prev_num_reg_images) {
                            snapshot_prev_num_reg_images = reconstruction.NumRegImages();
                            WriteSnapshot(reconstruction, options_->snapshot_path);
                        }
                        Callback(NEXT_IMAGE_REG_CALLBACK);
                        //break;
                    } else {
                        std::cout << "  => Could not register, trying another image."
                                  << std::endl;

                        // If initial pair fails to continue for some time,
                        // abort and try different initial pair.
                        const size_t kMinNumInitialRegTrials = 30;
                        if (reg_trial >= kMinNumInitialRegTrials &&
                            reconstruction.NumRegImages() <
                                    static_cast<size_t>(options_->min_model_size)) {
                            break;
                        }
                    }
                }
                //每处理两个相机组便做迭代rigba
                //std::cout << "reconstruction.NumRegImages()=" << reconstruction.NumRegImages() << std::endl;
                //IterativeRigLocalRefinement(*options_, &mapper, next_images[reconstruction.NumCameras() - 1], &camera_rigs);
                if (reconstruction.NumRegImages() % 12 == 0) {
                    IterativeRigGlobalRefinement(*options_, &mapper, &camera_rigs, "");
                    //ba_prev_num_points     = reconstruction.NumPoints3D();
                    //ba_prev_num_reg_images = reconstruction.NumRegImages();
                }
                //Callback(LAST_IMAGE_REG_CALLBACK);
                const size_t max_model_overlap =
                        static_cast<size_t>(options_->max_model_overlap);
                if (mapper.NumSharedRegImages() >= max_model_overlap) {
                    break;
                }

                // If no image could be registered, try a single final global iterative
                // bundle adjustment and try again to register one image. If this fails
                // once, then exit the incremental mapping.
                if (!reg_next_success && prev_reg_next_success) {
                    reg_next_success      = true;
                    prev_reg_next_success = false;
                    IterativeFrontWideGlobalRefinement(*options_, &mapper);
                } else {
                    prev_reg_next_success = reg_next_success;
                }
            }
            IterativeRigGlobalRefinement(*options_, &mapper, &camera_rigs, output_path);
            //ba_prev_num_points     = reconstruction.NumPoints3D();
            //ba_prev_num_reg_images = reconstruction.NumRegImages();
            std::cout << "all the images are handled" << std::endl;
            Callback(LAST_IMAGE_REG_CALLBACK);
            //mapper.EndReconstruction(kDiscardReconstruction);
            if (IsStopped()) {
                const bool kDiscardReconstruction = false;
                mapper.EndReconstruction(kDiscardReconstruction);
                break;
            }

            const size_t min_model_size =
                    std::min(database_cache_.NumImages(),
                             static_cast<size_t>(options_->min_model_size));
            if ((options_->multiple_models &&
                 reconstruction.NumRegImages() < min_model_size) ||
                reconstruction.NumRegImages() == 0) {
                mapper.EndReconstruction(kDiscardReconstruction);
                reconstruction_manager_->Delete(reconstruction_idx);
            } else {
                const bool kDiscardReconstruction = false;
                mapper.EndReconstruction(kDiscardReconstruction);
            }

            const size_t max_num_models = static_cast<size_t>(options_->max_num_models);
            if (initial_reconstruction_given || !options_->multiple_models ||
                reconstruction_manager_->Size() >= max_num_models ||
                mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {
                break;
            }
        }
    }

    void IncrementalMapperController::Reconstruct(
            const IncrementalMapper::Options &init_mapper_options) {
        const bool kDiscardReconstruction = true;

        //////////////////////////////////////////////////////////////////////////////
        // Main loop
        //////////////////////////////////////////////////////////////////////////////

        IncrementalMapper mapper(&database_cache_);

        // Is there a sub-model before we start the reconstruction? I.e. the user
        // has imported an existing reconstruction.
        const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
        CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                        "single reconstruction, but "
                                                        "multiple are given.";

        for (int num_trials = 0; num_trials < options_->init_num_trials;
             ++num_trials) {
            BlockIfPaused();
            if (IsStopped()) {
                break;
            }

            size_t reconstruction_idx;
            if (!initial_reconstruction_given || num_trials > 0) {
                reconstruction_idx = reconstruction_manager_->Add();
            } else {
                reconstruction_idx = 0;
            }

            Reconstruction &reconstruction =
                    reconstruction_manager_->Get(reconstruction_idx);

            mapper.BeginReconstruction(&reconstruction);

            ////////////////////////////////////////////////////////////////////////////
            // Register initial pair
            ////////////////////////////////////////////////////////////////////////////

            if (reconstruction.NumRegImages() == 0) {
                image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
                image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

                // Try to find good initial pair.
                if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
                    PrintHeading1("Finding good initial image pair");
                    const bool find_init_success = mapper.FindInitialFrontWideImagePair(
                            init_mapper_options, &image_id1, &image_id2);
                    if (!find_init_success) {
                        std::cout << "  => No good initial image pair found." << std::endl;
                        mapper.EndReconstruction(kDiscardReconstruction);
                        reconstruction_manager_->Delete(reconstruction_idx);
                        break;
                    }
                } else {
                    if (!reconstruction.ExistsImage(image_id1) ||
                        !reconstruction.ExistsImage(image_id2)) {
                        std::cout << StringPrintf(
                                             "  => Initial image pair #%d and #%d do not exist.",
                                             image_id1, image_id2)
                                  << std::endl;
                        mapper.EndReconstruction(kDiscardReconstruction);
                        reconstruction_manager_->Delete(reconstruction_idx);
                        return;
                    }
                }

                PrintHeading1(StringPrintf("Initializing with image pair #%d and #%d",
                                           image_id1, image_id2));
                const bool reg_init_success = mapper.RegisterInitialFrontWideImagePair(
                        init_mapper_options, image_id1, image_id2);
                if (!reg_init_success) {
                    std::cout << "  => Initialization failed - possible solutions:"
                              << std::endl
                              << "     - try to relax the initialization constraints"
                              << std::endl
                              << "     - manually select an initial image pair"
                              << std::endl;
                    mapper.EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    break;
                }

                AdjustGlobalBundle(*options_, &mapper);
                FilterFrontWidePoints(*options_, &mapper);
                FilterImages(*options_, &mapper);

                // Initial image pair failed to register.
                if (reconstruction.NumRegImages() == 0 ||
                    reconstruction.NumPoints3D() == 0) {
                    mapper.EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    // If both initial images are manually specified, there is no need for
                    // further initialization trials.
                    if (options_->init_image_id1 != -1 && options_->init_image_id2 != -1) {
                        break;
                    } else {
                        continue;
                    }
                }

                if (options_->extract_colors) {
                    ExtractColors(image_path_, image_id1, &reconstruction);
                }
            }

            Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

            ////////////////////////////////////////////////////////////////////////////
            // Incremental mapping
            ////////////////////////////////////////////////////////////////////////////

            size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
            size_t ba_prev_num_reg_images       = reconstruction.NumRegImages();
            size_t ba_prev_num_points           = reconstruction.NumPoints3D();

            bool reg_next_success      = true;
            bool prev_reg_next_success = true;
            while (reg_next_success) {
                BlockIfPaused();
                if (IsStopped()) {
                    break;
                }

                reg_next_success = false;

                const std::vector<image_t> next_images =
                        mapper.FindNextFrontWideImages(options_->Mapper());

                if (next_images.empty()) {
                    break;
                }

                for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
                    const image_t next_image_id = next_images[reg_trial];
                    const Image &next_image     = reconstruction.Image(next_image_id);

                    PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                               reconstruction.NumRegImages() + 1));

                    std::cout << StringPrintf("  => Image sees %d / %d points",
                                              next_image.NumVisiblePoints3D(),
                                              next_image.NumObservations())
                              << std::endl;

                    reg_next_success =
                            mapper.RegisterNextImage(options_->Mapper(), next_image_id);

                    if (reg_next_success) {
                        //std::cout<<"before triangulate, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                        TriangulateImage(*options_, next_image, &mapper);
                        //std::cout<<"after triangulate, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                        IterativeLocalRefinement(*options_, next_image_id, &mapper);
                        //std::cout<<"after local refinement, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                        if (reconstruction.NumRegImages() >=
                                    options_->ba_global_images_ratio * ba_prev_num_reg_images ||
                            reconstruction.NumRegImages() >=
                                    options_->ba_global_images_freq + ba_prev_num_reg_images ||
                            reconstruction.NumPoints3D() >=
                                    options_->ba_global_points_ratio * ba_prev_num_points ||
                            reconstruction.NumPoints3D() >=
                                    options_->ba_global_points_freq + ba_prev_num_points) {
                            //std::cout<<"before global ba, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                            IterativeGlobalRefinement(*options_, &mapper);
                            //std::cout<<"after global ba, reconstruction.NumPoints3D()="<<reconstruction.NumPoints3D()<<std::endl;
                            ba_prev_num_points     = reconstruction.NumPoints3D();
                            ba_prev_num_reg_images = reconstruction.NumRegImages();
                        }

                        if (options_->extract_colors) {
                            ExtractColors(image_path_, next_image_id, &reconstruction);
                        }

                        if (options_->snapshot_images_freq > 0 &&
                            reconstruction.NumRegImages() >=
                                    options_->snapshot_images_freq +
                                            snapshot_prev_num_reg_images) {
                            snapshot_prev_num_reg_images = reconstruction.NumRegImages();
                            WriteSnapshot(reconstruction, options_->snapshot_path);
                        }

                        Callback(NEXT_IMAGE_REG_CALLBACK);

                        break;
                    } else {
                        std::cout << "  => Could not register, trying another image."
                                  << std::endl;

                        // If initial pair fails to continue for some time,
                        // abort and try different initial pair.
                        const size_t kMinNumInitialRegTrials = 30;
                        if (reg_trial >= kMinNumInitialRegTrials &&
                            reconstruction.NumRegImages() <
                                    static_cast<size_t>(options_->min_model_size)) {
                            break;
                        }
                    }
                }

                const size_t max_model_overlap =
                        static_cast<size_t>(options_->max_model_overlap);
                if (mapper.NumSharedRegImages() >= max_model_overlap) {
                    break;
                }

                // If no image could be registered, try a single final global iterative
                // bundle adjustment and try again to register one image. If this fails
                // once, then exit the incremental mapping.
                if (!reg_next_success && prev_reg_next_success) {
                    reg_next_success      = true;
                    prev_reg_next_success = false;
                    IterativeGlobalRefinement(*options_, &mapper);
                } else {
                    prev_reg_next_success = reg_next_success;
                }
            }

            if (IsStopped()) {
                const bool kDiscardReconstruction = false;
                mapper.EndReconstruction(kDiscardReconstruction);
                break;
            }

            // Only run final global BA, if last incremental BA was not global.
            if (reconstruction.NumRegImages() >= 2 &&
                reconstruction.NumRegImages() != ba_prev_num_reg_images &&
                reconstruction.NumPoints3D() != ba_prev_num_points) {
                IterativeGlobalRefinement(*options_, &mapper);
            }

            // If the total number of images is small then do not enforce the minimum
            // model size so that we can reconstruct small image collections.
            const size_t min_model_size =
                    std::min(database_cache_.NumImages(),
                             static_cast<size_t>(options_->min_model_size));
            if ((options_->multiple_models &&
                 reconstruction.NumRegImages() < min_model_size) ||
                reconstruction.NumRegImages() == 0) {
                mapper.EndReconstruction(kDiscardReconstruction);
                reconstruction_manager_->Delete(reconstruction_idx);
            } else {
                const bool kDiscardReconstruction = false;
                mapper.EndReconstruction(kDiscardReconstruction);
            }

            Callback(LAST_IMAGE_REG_CALLBACK);

            const size_t max_num_models = static_cast<size_t>(options_->max_num_models);
            if (initial_reconstruction_given || !options_->multiple_models ||
                reconstruction_manager_->Size() >= max_num_models ||
                mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {
                break;
            }
        }
    }

}// namespace colmap
