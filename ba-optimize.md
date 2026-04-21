# 实现方案: 为RigReconstruct函数添加三个相机组优化策略

## 概述

本方案详细说明了如何在COLMAP的RigReconstruct函数中实现三个相机组优化策略:
1. **分步优化**: 固定外参BA后放开外参BA
2. **外参先验约束**: 对车体到相机的外参加上高斯先验
3. **平移和旋转解耦**: 只优化旋转外参,固定平移外参

---

## 1. 分步优化实现方案

### 实现原理
将BA循环分为两个阶段:
- **第一阶段**: 固定相机相对外参(设为常量),只优化车体位姿和3D点
- **第二阶段**: 放开相机相对外参的自由度,进行全局BA微调

### 实现位置

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/controllers/incremental_mapper.cc`

**函数**: `IterativeRigGlobalRefinement` (第312-469行)

### 具体实现步骤

#### 步骤1.1: 修改BA循环结构

**当前代码** (第319行):
```cpp
for (int i = 0; i < options.ba_global_max_refinements - 2; ++i) {
```

**修改为**:
```cpp
// 第一阶段: 固定外参,优化轨迹和点 (前半部分迭代)
int first_stage_iterations = (options.ba_global_max_refinements - 2) / 2;
for (int i = 0; i < first_stage_iterations; ++i) {
    // 设置标志: 固定相对位姿
    bool estimate_rig_relative_poses = false;
    
    // ... 现有代码 ...
}

// 第二阶段: 放开外参,全局BA微调 (后半部分迭代)
int second_stage_iterations = (options.ba_global_max_refinements - 2) - first_stage_iterations;
for (int i = 0; i < second_stage_iterations; ++i) {
    // 设置标志: 优化相对位姿
    bool estimate_rig_relative_poses = true;
    
    // ... 现有代码 ...
}
```

#### 步骤1.2: 使用refine_relative_poses标志

**当前代码** (第399行):
```cpp
bool estimate_rig_relative_poses = true;  // 是否估计相机组的相对位姿
```

**修改为**: 根据阶段动态设置
```cpp
// 在每个阶段循环内设置
bool estimate_rig_relative_poses = (i >= first_stage_iterations);
```

**应用标志** (第400-428行):
```cpp
RigBundleAdjuster::Options rig_ba_options;
rig_ba_options.refine_relative_poses = estimate_rig_relative_poses;  // 使用阶段标志

OptionManager my_options;
BundleAdjustmentConfig config;

// ... 现有代码 ...

RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);
CHECK(bundle_adjuster.Solve(&(mapper->MyGetReconstruction()), camera_rigs));
```

### 关键点说明

1. **分割迭代次数**: 将`ba_global_max_refinements - 2`次迭代平均分成两部分
   - 第一部分: 固定外参优化
   - 第二部分: 放开外参优化

2. **利用现有机制**: `RigBundleAdjuster::Options`已经有`refine_relative_poses`选项
   - 设为`false`: 外参固定 (第1396-1400行会设置qvec和tvec为常量)
   - 设为`true`: 外参可优化

3. **影响范围**: 这个修改只影响`IterativeRigGlobalRefinement`函数,不改变其他BA调用

---

## 2. 外参先验约束实现方案

### 实现原理
为车体到相机的相对外参(旋转和平移)添加高斯先验约束,限制其在小范围内微调:
- 平移: 几厘米范围内
- 旋转: 零点几度范围内

### 实现位置

**主要文件**:
1. `/Users/yg/work/MRASfM/colmap-3.8/src/base/cost_functions.h` - 定义新的CostFunction
2. `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.h` - 添加选项
3. `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.cc` - 添加先验约束

### 具体实现步骤

#### 步骤2.1: 定义相对位姿先验CostFunction

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/base/cost_functions.h`

**位置**: 在`PoseConstraintCostFunction`定义后 (约第441行后)

**添加代码**:
```cpp
// 相对位姿先验约束CostFunction
// 用于约束车体到相机的相对外参在小范围内变化
struct RelativePosePriorCostFunction {
    Eigen::Vector4d prior_qvec_;  // 先验旋转(四元数)
    Eigen::Vector3d prior_tvec_;  // 先验平移
    Eigen::Vector3d tvec_weight_; // 平移权重(控制允许的偏差范围)
    double qvec_weight_;          // 旋转权重(控制允许的偏差范围)

    RelativePosePriorCostFunction(const Eigen::Vector4d& qvec_prior, 
                                  const Eigen::Vector3d& tvec_prior,
                                  const Eigen::Vector3d& tvec_weight,
                                  double qvec_weight)
        : prior_qvec_(qvec_prior), 
          prior_tvec_(tvec_prior),
          tvec_weight_(tvec_weight),
          qvec_weight_(qvec_weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec_prior,
                                       const Eigen::Vector3d& tvec_prior,
                                       const Eigen::Vector3d& tvec_weight,
                                       double qvec_weight) {
        // 残差维度: 3(平移) + 1(旋转角度) = 4
        // 参数块: qvec(4) + tvec(3)
        return (new ceres::AutoDiffCostFunction<RelativePosePriorCostFunction, 4, 4, 3>(
                new RelativePosePriorCostFunction(qvec_prior, tvec_prior, 
                                                  tvec_weight, qvec_weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        
        // 计算平移误差 (当前值 - 先验值)
        Vec3T t_error;
        t_error(0) = T(tvec[0]) - T(prior_tvec_(0));
        t_error(1) = T(tvec[1]) - T(prior_tvec_(1));
        t_error(2) = T(tvec[2]) - T(prior_tvec_(2));
        
        // 计算旋转误差 (角度差)
        const Eigen::Quaternion<T> q_current(qvec[0], qvec[1], qvec[2], qvec[3]);
        const Eigen::Quaternion<T> q_prior(
            T(prior_qvec_(0)), T(prior_qvec_(1)), 
            T(prior_qvec_(2)), T(prior_qvec_(3))
        );
        
        // 计算四元数之间的差
        Eigen::Quaternion<T> q_diff = q_current.conjugate() * q_prior;
        
        // 将四元数差转换为角度轴形式
        Eigen::Matrix<T, 3, 1> angle_axis;
        angle_axis = safe_quat2axis(q_diff);
        
        // 加权残差
        residuals[0] = tvec_weight_(0) * t_error(0);
        residuals[1] = tvec_weight_(1) * t_error(1);
        residuals[2] = tvec_weight_(2) * t_error(2);
        residuals[3] = qvec_weight_ * angle_axis.norm();  // 旋转角度的范数
        
        return true;
    }
};
```

#### 步骤2.2: 添加先验约束选项

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.h`

**位置**: 在`RigBundleAdjuster::Options`结构体中 (第290-300行)

**修改当前结构体**:
```cpp
struct Options {
    // Whether to optimize the relative poses of the camera rigs.
    bool refine_relative_poses = true;

    // The maximum allowed reprojection error for an observation to be
    // considered in the bundle adjustment.
    double max_reproj_error = 1000.0;

    // ========== 新增: 相对位姿先验约束选项 ==========
    // 是否启用相对位姿先验约束
    bool use_relative_pose_prior = false;
    
    // 相对平移的权重 (控制允许的偏差范围, 值越大约束越强)
    // 例如: (100, 100, 100) 表示允许约1cm的偏差 (1/100米)
    Eigen::Vector3d relative_tvec_weight = Eigen::Vector3d(100.0, 100.0, 100.0);
    
    // 相对旋转的权重 (控制允许的角度偏差, 值越大约束越强)
    // 例如: 1000 表示允许约0.06度的偏差 (1/1000弧度)
    double relative_qvec_weight = 1000.0;
};
```

#### 步骤2.3: 在BA中添加先验约束

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.cc`

**位置**: 在`RigBundleAdjuster::SetUp`函数中 (第1043-1063行)

**修改** - 在调用`ParameterizeCameraRigs`之前添加先验约束:
```cpp
void RigBundleAdjuster::SetUp(Reconstruction *reconstruction,
                              std::vector<CameraRig> *camera_rigs,
                              ceres::LossFunction *loss_function) {
    ComputeCameraRigPoses(*reconstruction, *camera_rigs);
    for (const image_t image_id: config_.Images()) {
        AddImageToProblem(image_id, reconstruction, camera_rigs, loss_function);
    }
    for (const auto point3D_id: config_.VariablePoints()) {
        std::cerr << "variable_point3D_id:" << point3D_id << std::endl;
        AddPointToProblem(point3D_id, reconstruction, loss_function);
    }
    for (const auto point3D_id: config_.ConstantPoints()) {
        std::cerr << "constant_point3D_id:" << point3D_id << std::endl;
        AddPointToProblem(point3D_id, reconstruction, loss_function);
    }

    // ========== 新增: 添加相对位姿先验约束 ==========
    if (rig_options_.use_relative_pose_prior) {
        AddRelativePosePriors(camera_rigs, loss_function);
    }

    ParameterizeCameras(reconstruction);
    ParameterizePoints(reconstruction);
    ParameterizeCameraRigs(reconstruction);
}
```

**位置**: 在`RigBundleAdjuster`类中添加新函数 (在`SetUp`函数后,约第1063行后)

**添加新函数**:
```cpp
void RigBundleAdjuster::AddRelativePosePriors(
        std::vector<CameraRig> *camera_rigs,
        ceres::LossFunction *loss_function) {
    
    // 遍历所有相机组
    for (auto &camera_rig : *camera_rigs) {
        // 获取该相机组中所有相机的ID
        const auto camera_ids = camera_rig.GetCameraIds();
        
        // 对每个非参考相机添加先验约束
        for (const camera_t camera_id : camera_ids) {
            // 跳过参考相机 (参考相机的相对位姿是单位变换)
            if (camera_id == camera_rig.RefCameraId()) {
                continue;
            }
            
            // 获取该相机的相对位姿 (作为先验)
            const Eigen::Vector4d prior_qvec = camera_rig.RelativeQvec(camera_id);
            const Eigen::Vector3d prior_tvec = camera_rig.RelativeTvec(camera_id);
            
            // 获取相对位姿的参数块指针
            double *qvec_data = camera_rig.RelativeQvec(camera_id).data();
            double *tvec_data = camera_rig.RelativeTvec(camera_id).data();
            
            // 创建先验约束CostFunction
            ceres::CostFunction *prior_cost_function = 
                RelativePosePriorCostFunction::Create(
                    prior_qvec, prior_tvec,
                    rig_options_.relative_tvec_weight,
                    rig_options_.relative_qvec_weight);
            
            // 添加到优化问题 (不使用鲁棒损失函数, 因为先验应该是硬约束)
            problem_->AddResidualBlock(
                prior_cost_function,
                nullptr,  // 不使用损失函数,先验是硬约束
                qvec_data,
                tvec_data);
        }
    }
}
```

**位置**: 在`RigBundleAdjuster`类声明中添加函数声明

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.h`

**位置**: 在`private`部分 (约第312-330行)

**添加声明**:
```cpp
private:
    void SetUp(Reconstruction *reconstruction,
               std::vector<CameraRig> *camera_rigs,
               ceres::LossFunction *loss_function);
    void TearDown(Reconstruction *reconstruction,
                  const std::vector<CameraRig> &camera_rigs);

    // ========== 新增: 添加相对位姿先验约束 ==========
    void AddRelativePosePriors(std::vector<CameraRig> *camera_rigs,
                               ceres::LossFunction *loss_function);

    void AddImageToProblem(const image_t image_id, Reconstruction *reconstruction,
                           std::vector<CameraRig> *camera_rigs,
                           ceres::LossFunction *loss_function);
    
    // ... 其余现有声明 ...
```

### 权重参数说明

**平移权重** (`relative_tvec_weight`):
- 权重 = 100 表示允许约1cm偏差 (1/100米)
- 权重 = 50 表示允许约2cm偏差
- 建议范围: 50-200

**旋转权重** (`relative_qvec_weight`):
- 权重 = 1000 表示允许约0.06度偏差 (1/1000弧度)
- 权重 = 500 表示允许约0.11度偏差
- 建议范围: 500-2000

### 使用方法

在`IterativeRigGlobalRefinement`函数中:
```cpp
RigBundleAdjuster::Options rig_ba_options;

// 启用相对位姿先验约束
rig_ba_options.use_relative_pose_prior = true;
rig_ba_options.relative_tvec_weight = Eigen::Vector3d(100.0, 100.0, 100.0);  // 1cm
rig_ba_options.relative_qvec_weight = 1000.0;  // ~0.06度

RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);
```

---

## 3. 平移和旋转解耦实现方案

### 实现原理
只优化车体到相机的旋转外参(qvec),固定平移外参(tvec)为常量。这样可以减少优化变量,提高稳定性。

### 实现位置

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.h` 和 `bundle_adjustment.cc`

### 具体实现步骤

#### 步骤3.1: 添加解耦选项

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.h`

**位置**: 在`RigBundleAdjuster::Options`结构体中 (第290-300行附近)

**添加选项**:
```cpp
struct Options {
    // Whether to optimize the relative poses of the camera rigs.
    bool refine_relative_poses = true;

    // The maximum allowed reprojection error for an observation to be
    // considered in the bundle adjustment.
    double max_reproj_error = 1000.0;

    // ========== 新增: 平移和旋转解耦选项 ==========
    // 是否只优化相对旋转,固定相对平移
    bool refine_relative_rotation_only = false;
};
```

#### 步骤3.2: 修改参数固定逻辑

**文件**: `/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.cc`

**位置**: `RigBundleAdjuster::AddImageToProblem`函数 (第1389-1409行)

**当前代码**:
```cpp
// 如果观测数量大于0
if (num_observations > 0) {
    if (camera_rig != nullptr && !constant_pose) {
        parameterized_qvec_data_.insert(qvec_data);
        parameterized_qvec_data_.insert(rig_qvec_data);

        // 如果相对姿态的优化被禁用或者当前图像是参考相机,将相机姿态参数块设置为常量
        if (!rig_options_.refine_relative_poses ||
            image.CameraId() == camera_rig->RefCameraId()) {
            problem_->SetParameterBlockConstant(qvec_data);
        }
        problem_->SetParameterBlockConstant(tvec_data);
    }
    // ...
}
```

**修改为**:
```cpp
// 如果观测数量大于0
if (num_observations > 0) {
    if (camera_rig != nullptr && !constant_pose) {
        parameterized_qvec_data_.insert(qvec_data);
        parameterized_qvec_data_.insert(rig_qvec_data);

        // 如果相对姿态的优化被禁用或者当前图像是参考相机,将相机姿态参数块设置为常量
        if (!rig_options_.refine_relative_poses ||
            image.CameraId() == camera_rig->RefCameraId()) {
            problem_->SetParameterBlockConstant(qvec_data);
        }
        
        // ========== 修改: 根据解耦选项决定是否固定平移 ==========
        if (!rig_options_.refine_relative_poses || 
            rig_options_.refine_relative_rotation_only) {
            // 如果不优化相对位姿,或者只优化旋转,则固定平移
            problem_->SetParameterBlockConstant(tvec_data);
        }
    }
    // ...
}
```

### 关键点说明

1. **条件判断**: 只有当`refine_relative_poses=true`且`refine_relative_rotation_only=false`时,才优化平移
2. **默认行为**: 当`refine_relative_rotation_only=false`时,保持原有行为(优化旋转和平移)
3. **与先验约束的兼容**: 此选项与策略2(先验约束)可以同时使用
   - 如果同时启用,先验约束会作用于可优化的旋转参数

### 使用方法

在`IterativeRigGlobalRefinement`函数中:
```cpp
RigBundleAdjuster::Options rig_ba_options;

// 只优化旋转,固定平移
rig_ba_options.refine_relative_poses = true;
rig_ba_options.refine_relative_rotation_only = true;

RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);
```

---

## 4. 综合使用示例

### 示例1: 分步优化 + 先验约束

在`IterativeRigGlobalRefinement`函数中:
```cpp
int first_stage_iterations = (options.ba_global_max_refinements - 2) / 2;
int second_stage_iterations = (options.ba_global_max_refinements - 2) - first_stage_iterations;

// 第一阶段: 固定外参
for (int i = 0; i < first_stage_iterations; ++i) {
    RigBundleAdjuster::Options rig_ba_options;
    rig_ba_options.refine_relative_poses = false;  // 固定外参
    rig_ba_options.use_relative_pose_prior = false;
    
    // ... BA求解 ...
}

// 第二阶段: 放开外参 + 先验约束
for (int i = 0; i < second_stage_iterations; ++i) {
    RigBundleAdjuster::Options rig_ba_options;
    rig_ba_options.refine_relative_poses = true;  // 优化外参
    rig_ba_options.use_relative_pose_prior = true;  // 启用先验
    rig_ba_options.relative_tvec_weight = Eigen::Vector3d(100.0, 100.0, 100.0);
    rig_ba_options.relative_qvec_weight = 1000.0;
    
    // ... BA求解 ...
}
```

### 示例2: 分步优化 + 先验约束 + 旋转解耦

```cpp
// 第二阶段: 放开外参 + 先验约束 + 只优化旋转
for (int i = 0; i < second_stage_iterations; ++i) {
    RigBundleAdjuster::Options rig_ba_options;
    rig_ba_options.refine_relative_poses = true;
    rig_ba_options.refine_relative_rotation_only = true;  // 只优化旋转
    rig_ba_options.use_relative_pose_prior = true;
    rig_ba_options.relative_tvec_weight = Eigen::Vector3d(100.0, 100.0, 100.0);
    rig_ba_options.relative_qvec_weight = 1000.0;
    
    // ... BA求解 ...
}
```

---

## 5. 文件修改汇总

### 需要修改的文件

1. **`/Users/yg/work/MRASfM/colmap-3.8/src/controllers/incremental_mapper.cc`**
   - 修改`IterativeRigGlobalRefinement`函数 (第312-469行)
   - 实现分步优化逻辑

2. **`/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.h`**
   - 修改`RigBundleAdjuster::Options`结构体 (第290-300行)
   - 添加三个新选项和函数声明

3. **`/Users/yg/work/MRASfM/colmap-3.8/src/optim/bundle_adjustment.cc`**
   - 修改`RigBundleAdjuster::SetUp`函数 (第1043-1063行)
   - 修改`RigBundleAdjuster::AddImageToProblem`函数 (第1389-1409行)
   - 添加`AddRelativePosePriors`新函数

4. **`/Users/yg/work/MRASfM/colmap-3.8/src/base/cost_functions.h`**
   - 添加`RelativePosePriorCostFunction`结构体 (约第441行后)

### 修改行数统计

- `incremental_mapper.cc`: 约30行修改
- `bundle_adjustment.h`: 约20行修改
- `bundle_adjustment.cc`: 约80行修改(含新函数)
- `cost_functions.h`: 约60行新增

**总计**: 约190行代码修改/新增

---

## 6. 测试建议

### 测试场景1: 分步优化
- 对比启用/禁用分步优化的结果
- 检查外参的收敛过程是否更稳定

### 测试场景2: 先验约束
- 使用不同的权重参数测试
- 验证外参变化范围是否符合预期
- 检查是否提高了重建精度

### 测试场景3: 旋转解耦
- 对比优化旋转+平移 vs 只优化旋转
- 验证平移参数是否保持不变
- 检查是否提高了优化稳定性

### 测试场景4: 组合策略
- 测试分步+先验的组合
- 测试分步+解耦的组合
- 测试三者同时使用

---

## 7. 参数调优建议

### 分步优化
- 建议第一部分迭代占总迭代的30-50%
- 例如: 如果`ba_global_max_refinements=5`,则前1-2次固定外参,后3次放开

### 先验约束权重
**平移权重** (`relative_tvec_weight`):
- 严格约束: 200-500 (允许2-5mm偏差)
- 中等约束: 50-100 (允许1-2cm偏差)
- 宽松约束: 10-20 (允许5-10cm偏差)

**旋转权重** (`relative_qvec_weight`):
- 严格约束: 2000-5000 (允许0.01-0.03度偏差)
- 中等约束: 500-1000 (允许0.06-0.11度偏差)
- 宽松约束: 100-200 (允许0.3-0.6度偏差)

### 建议初始配置
```cpp
rig_ba_options.relative_tvec_weight = Eigen::Vector3d(100.0, 100.0, 100.0);
rig_ba_options.relative_qvec_weight = 1000.0;
```

---

## 8. 注意事项

1. **与现有代码的兼容性**
   - 所有新选项都有默认值,不影响现有代码行为
   - 可以通过配置文件或命令行参数控制

2. **性能影响**
   - 先验约束会添加少量残差块,对性能影响很小
   - 分步优化不会增加总迭代次数,只是重新分配

3. **数值稳定性**
   - 先验约束使用高斯先验,数值稳定
   - 权重不宜过大,否则可能导致数值问题

4. **参考相机处理**
   - 参考相机的相对位姿是单位变换,不应添加先验约束
   - 代码中已处理跳过参考相机

5. **与ENABLE_POSITION_PRIOR的兼容**
   - 策略2(外参先验)与现有的`ENABLE_POSITION_PRIOR`不同
   - `ENABLE_POSITION_PRIOR`约束的是车体位姿(rig pose)
   - 新的先验约束的是相机相对外参(relative pose)

---

## 9. 实现顺序建议

建议按以下顺序实现:

1. **优先级1: 策略1(分步优化)**
   - 修改最少,风险最低
   - 可以独立测试效果

2. **优先级2: 策略3(旋转解耦)**
   - 实现简单,易于验证
   - 可以快速看到效果

3. **优先级3: 策略2(先验约束)**
   - 需要定义新的CostFunction
   - 需要仔细调优权重参数

每个策略实现后独立测试,确认无误后再实现下一个。

---

## 10. 总结

本方案提供了三个相机组优化策略的详细实现方案:

1. **分步优化**: 通过分割BA循环,先固定外参优化轨迹和点,再放开外参微调
2. **外参先验约束**: 定义新的`RelativePosePriorCostFunction`,为相机相对外参添加高斯先验
3. **平移和旋转解耦**: 添加`refine_relative_rotation_only`选项,只优化旋转外参

三个策略可以独立使用,也可以组合使用,为不同的应用场景提供灵活的优化策略选择。
