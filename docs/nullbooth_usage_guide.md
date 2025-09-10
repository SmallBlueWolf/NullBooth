# NullBooth 使用指南

## 概述

NullBooth 是基于 AlphaEdit 论文的 Null-Space Constrained Knowledge Editing 方法在 Diffusion 模型上的实现。该系统允许在保持原有知识的同时对模型进行精确编辑。

## 快速开始

### 1. 环境配置

确保你已经安装了必要的依赖：

```bash
pip install torch torchvision diffusers transformers accelerate matplotlib seaborn tqdm numpy
```

### 2. 配置文件设置

在 `configs/nullbooth.yaml` 中启用 NullBooth 模式：

```yaml
nullbooth:
  enable: true  # 启用 NullBooth 模式
  original_knowledge_prompts: "./dataset/dog_prompts_1000.txt"
  cov_matrices_output_dir: "./cov_matrices" 
  visual_attention_map: true  # 可选：生成注意力图可视化
  num_denoising_steps: 50
  nullspace_threshold: 2e-2
  collect_features:
    q_features: true
    k_features: true  
    v_features: true
    out_features: true
  cross_attention_layers: "all"
```

### 3. 构建协方差矩阵

运行 `build_cov.py` 脚本来计算协方差矩阵：

```bash
# 使用默认配置
python build_cov.py

# 或指定配置文件
python build_cov.py --config configs/nullbooth.yaml
```

这个过程将：
- 加载指定的 Diffusion 模型
- 读取原始知识 prompts 文件
- 对每个 prompt 进行完整的去噪过程（50步）
- 收集所有 cross-attention 层的 Q/K/V/OUT 特征
- 计算协方差矩阵和 null-space 投影矩阵
- 保存结果到指定目录

### 4. 输出结构

运行完成后，你会得到以下目录结构：

```
cov_matrices/
└── run_20231201_143022/  # 时间戳目录
    ├── step_000/         # 每个去噪步骤
    ├── step_001/
    ├── ...
    ├── step_049/
    └── visualizations/   # 可视化结果（如果启用）
```

## 详细配置说明

### 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable` | 启用/禁用 NullBooth 模式 | `false` |
| `original_knowledge_prompts` | 原始知识 prompts 文件路径 | `"./dataset/dog_prompts_1000.txt"` |
| `cov_matrices_output_dir` | 协方差矩阵输出目录 | `"./cov_matrices"` |
| `nullspace_threshold` | 零空间阈值 | `2e-2` |
| `num_denoising_steps` | 去噪步数 | `50` |

### 特征收集配置

```yaml
collect_features:
  q_features: true   # 收集 Query 特征
  k_features: true   # 收集 Key 特征
  v_features: true   # 收集 Value 特征 
  out_features: true # 收集 Output 特征
```

### 可视化配置

```yaml
visual_attention_map: true  # 生成注意力图热力图
```

启用后会生成：
- 每个时间步的注意力图热力图
- 协方差矩阵统计分析图
- 零空间维度分布图

## 原始知识 Prompts 文件格式

`dog_prompts_1000.txt` 文件格式：

```
# 这是注释行，会被忽略
a photo of a dog
a beautiful landscape
a cat sitting on a chair
# 可以添加更多注释
a car driving on a road
...
```

- 每行一个 prompt
- 以 `#` 开头的行被视为注释
- 空行会被忽略
- 建议使用多样化的 prompts 以获得更好的协方差矩阵

## 高级用法

### 1. 自定义层选择

如果只想收集特定层的特征：

```yaml
cross_attention_layers: [0, 1, 2, 5, 10]  # 指定层索引
```

### 2. 批处理大文件

对于大型 prompts 文件，脚本会自动进行内存管理，但你可以通过修改以下参数优化性能：

```python
# 在 build_cov.py 中
BATCH_SIZE = 10  # 批处理大小
MAX_MEMORY_USAGE = 0.8  # 最大内存使用率
```

### 3. 分布式计算

对于大规模计算，可以将 prompts 文件分割并在多个 GPU 上并行运行：

```bash
# 分割 prompts 文件
split -l 250 dataset/dog_prompts_1000.txt prompts_part_

# 在不同 GPU 上运行
CUDA_VISIBLE_DEVICES=0 python build_cov.py --config config_part1.yaml &
CUDA_VISIBLE_DEVICES=1 python build_cov.py --config config_part2.yaml &
```

## 故障排除

### 常见问题

1. **内存不足错误**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方案：
   - 减少 `num_denoising_steps`
   - 设置 `mixed_precision: "fp16"`
   - 减少并行处理的 prompts 数量

2. **文件未找到错误**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: './dataset/dog_prompts_1000.txt'
   ```
   解决方案：
   - 检查 prompts 文件路径是否正确
   - 确保文件存在且可读

3. **模型加载失败**
   ```
   OSError: Model not found
   ```
   解决方案：
   - 检查 `pretrained_model_name_or_path` 配置
   - 确保有足够的网络连接下载模型
   - 考虑使用本地模型路径

### 性能优化建议

1. **使用混合精度**：
   ```yaml
   mixed_precision: "fp16"  # 或 "bf16"
   ```

2. **启用内存优化**：
   ```yaml
   enable_xformers_memory_efficient_attention: true
   ```

3. **调整批处理大小**：
   ```python
   # 根据 GPU 内存调整
   train_batch_size: 1  # 对于协方差计算，通常设为 1
   ```

## 输出数据使用

### 加载协方差矩阵

```python
import numpy as np
import json

def load_covariance_data(run_dir, timestep, layer_name, feature_type):
    """加载特定的协方差矩阵数据"""
    step_dir = f"{run_dir}/step_{timestep:03d}"
    layer_dir = f"{step_dir}/{layer_name.replace('.', '_')}"
    
    # 加载协方差矩阵
    cov_matrix = np.load(f"{layer_dir}/{feature_type}_covariance.npy")
    
    # 加载投影矩阵（如果存在）
    proj_path = f"{layer_dir}/{feature_type}_projection.npy"
    projection_matrix = np.load(proj_path) if os.path.exists(proj_path) else None
    
    # 加载元数据
    with open(f"{layer_dir}/{feature_type}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return cov_matrix, projection_matrix, metadata
```

### 应用零空间约束

```python
def apply_nullspace_constraint(delta, projection_matrix):
    """应用零空间约束到参数扰动"""
    if projection_matrix is not None:
        # 投影到零空间：Δ' = Δ @ P
        constrained_delta = delta @ projection_matrix
        return constrained_delta
    else:
        return delta  # 无零空间约束
```

## 理论背景

NullBooth 基于 AlphaEdit 论文的核心思想：

1. **问题**：传统的模型编辑会破坏原有知识
2. **解决方案**：将参数扰动投影到原有知识的零空间
3. **数学基础**：如果 `Δ'·K₀ = 0`，那么 `(W + Δ')K₀ = WK₀ = V₀`
4. **实现**：通过 SVD 分解协方差矩阵 `K₀K₀ᵀ` 找到零空间

关键公式：
- 协方差矩阵：`C = K₀K₀ᵀ`
- SVD 分解：`C = UΣVᵀ`
- 零空间投影：`P = ŮŮᵀ`（Ů 对应小特征值的特征向量）
- 约束编辑：`Δ' = ΔP`

## 下一步

完成协方差矩阵构建后，你可以：

1. 分析零空间结构和分布
2. 实现基于零空间约束的 DreamBooth 训练
3. 评估知识保持效果
4. 与传统方法进行对比分析

更多详细信息请参考：
- [协方差矩阵存储逻辑文档](docs/covariance_matrix_storage_logic.md)
- AlphaEdit 原始论文：[链接]
- 项目代码仓库：[链接]