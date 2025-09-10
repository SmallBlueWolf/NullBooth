# NullBooth 协方差矩阵存储逻辑文档

## 概述

本文档详细描述了 NullBooth 项目中协方差矩阵的计算、存储和组织逻辑，基于 AlphaEdit 论文中的 Null-Space Constrained Knowledge Editing 方法。

## 理论基础

### AlphaEdit 方法核心

1. **原有知识矩阵 K₀**: 通过大量样本的特征收集构建
2. **协方差矩阵**: `K₀K₀ᵀ ∈ R^(d₀×d₀)`，其中 d₀ 是特征维度
3. **Null Space 投影**: 通过 SVD 分解找到零空间，构建投影矩阵 `P = ŮŮᵀ`
4. **约束编辑**: 将参数扰动投影到原有知识的零空间中

### Diffusion 模型中的应用

在 Diffusion 模型中，我们将 AlphaEdit 的概念扩展到：
- **Cross-Attention 层**: 文本-图像交互的关键层
- **Q/K/V/OUT 特征**: 注意力机制的四个关键组件
- **时间步采样**: 从完整的噪声范围（0-1000）中采样指定数量的时间步，类似现代采样器的快速推理策略

## 数据收集流程

### 1. 特征收集阶段

```python
# 时间步采样策略
total_timesteps = 1000  # Stable Diffusion 的总时间步数
num_sample_steps = 50   # 配置中指定的采样数量

# 计算采样间隔
step_interval = total_timesteps // num_sample_steps  # 1000 // 50 = 20
sampled_timesteps = [0, 20, 40, 60, ..., 980]  # 50个均匀分布的时间步

# 对于每个采样时间步和每个 prompt
for timestep in sampled_timesteps:
    for prompt in original_knowledge_prompts:
        # 收集所有 cross-attention 层的 Q, K, V, OUT 特征
        features[f"timestep_{timestep:04d}"][layer_name] = {
            'q': query_features,      # Query 特征
            'k': key_features,        # Key 特征  
            'v': value_features,      # Value 特征
            'out': output_features    # Output 特征
        }
```

### 2. 协方差矩阵计算

```python
# 对于每种特征类型
for feature_type in ['q', 'k', 'v', 'out']:
    # 1. 特征预处理
    features = reshape_and_center(collected_features)
    
    # 2. 计算协方差矩阵 (AlphaEdit 中的 K₀K₀ᵀ)
    cov_matrix = torch.mm(features.T, features) / (n_samples - 1)
    
    # 3. SVD 分解
    U, S, _ = torch.linalg.svd(cov_matrix, full_matrices=False)
    
    # 4. 找到零空间 (特征值 < threshold)
    null_indices = (S < nullspace_threshold).nonzero()
    
    # 5. 构建投影矩阵 P = ŮŮᵀ
    U_null = U[:, null_indices]
    projection_matrix = torch.mm(U_null, U_null.T)
```

## 文件组织结构

### 目录层次结构

```
cov_matrices/
├── run_YYYYMMDD_HHMMSS/           # 时间戳运行目录
│   ├── timestep_0980/             # 实际时间步 980 (高噪声)
│   │   ├── down_blocks_0_attentions_0_transformer_blocks_0_attn2/
│   │   │   ├── q_covariance.npy   # Query 协方差矩阵
│   │   │   ├── q_projection.npy   # Query 投影矩阵
│   │   │   ├── q_metadata.json    # Query 元数据
│   │   │   ├── k_covariance.npy   # Key 协方差矩阵
│   │   │   ├── k_projection.npy   # Key 投影矩阵
│   │   │   ├── k_metadata.json    # Key 元数据
│   │   │   ├── v_covariance.npy   # Value 协方差矩阵
│   │   │   ├── v_projection.npy   # Value 投影矩阵
│   │   │   ├── v_metadata.json    # Value 元数据
│   │   │   ├── out_covariance.npy # Output 协方差矩阵
│   │   │   ├── out_projection.npy # Output 投影矩阵
│   │   │   └── out_metadata.json  # Output 元数据
│   │   ├── down_blocks_0_attentions_0_transformer_blocks_1_attn2/
│   │   └── ... (其他 cross-attention 层)
│   ├── timestep_0960/             # 实际时间步 960
│   ├── timestep_0940/             # 实际时间步 940
│   ├── ...
│   ├── timestep_0020/             # 实际时间步 20 (低噪声)
│   └── visualizations/            # 可视化文件 (可选)
│       ├── timestep_0980/
│       │   ├── attention_layer1.png
│       │   └── attention_layer2.png
│       ├── timestep_0960/
│       └── covariance_summaries/
│           └── covariance_statistics.png
```

### 文件格式说明

#### 1. 协方差矩阵文件 (`*_covariance.npy`)

- **格式**: NumPy 数组格式 (.npy)
- **数据类型**: float32
- **形状**: `(feature_dim, feature_dim)`
- **内容**: 协方差矩阵 K₀K₀ᵀ

#### 2. 投影矩阵文件 (`*_projection.npy`)

- **格式**: NumPy 数组格式 (.npy) 
- **数据类型**: float32
- **形状**: `(feature_dim, feature_dim)`
- **内容**: 零空间投影矩阵 P = ŮŮᵀ
- **注意**: 如果没有零空间 (null_space_dim=0)，则不保存此文件

#### 3. 元数据文件 (`*_metadata.json`)

```json
{
  "original_shape": [1, 8, 4096, 320],    // 原始特征形状
  "n_samples": 1000,                      // 样本数量
  "feature_dim": 320,                     // 特征维度
  "null_space_dim": 45,                   // 零空间维度
  "singular_values": [12.34, 8.76, ...], // 奇异值数组
  "nullspace_threshold": 0.02             // 零空间阈值
}
```

## 数据加载和使用

### 加载协方差矩阵

```python
import numpy as np
import json

def load_covariance_data(timestep, layer_name, feature_type, base_dir):
    """加载特定层和特征类型的协方差数据"""
    
    # 构建路径
    timestep_dir = f"{base_dir}/timestep_{timestep:04d}"
    layer_dir = f"{timestep_dir}/{layer_name.replace('.', '_')}"
    
    # 加载协方差矩阵
    cov_path = f"{layer_dir}/{feature_type}_covariance.npy"
    covariance_matrix = np.load(cov_path)
    
    # 加载投影矩阵 (如果存在)
    proj_path = f"{layer_dir}/{feature_type}_projection.npy"
    projection_matrix = np.load(proj_path) if os.path.exists(proj_path) else None
    
    # 加载元数据
    meta_path = f"{layer_dir}/{feature_type}_metadata.json"
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'covariance': covariance_matrix,
        'projection': projection_matrix,
        'metadata': metadata
    }
```

### 使用示例

```python
# 加载特定时间步和层的数据
timestep = 500  # 实际去噪时间步值 (0-1000范围内)
layer_name = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"
feature_type = "q"  # Query 特征

data = load_covariance_data(timestep, layer_name, feature_type, "cov_matrices/run_20231201_143022")

# 获取投影矩阵用于约束编辑
P = data['projection']
if P is not None:
    # 应用零空间投影: Δ' = Δ @ P
    constrained_delta = original_delta @ P
```

## 配置参数说明

### NullBooth 配置块

```yaml
nullbooth:
  enable: true                                    # 启用 NullBooth 模式
  original_knowledge_prompts: "./dataset/dog_prompts_1000.txt"  # 原始知识 prompts 文件
  cov_matrices_output_dir: "./cov_matrices"      # 协方差矩阵输出目录
  visual_attention_map: true                     # 保存注意力图可视化
  num_denoising_steps: 50                        # 去噪步数
  nullspace_threshold: 2e-2                     # 零空间阈值 (与 AlphaEdit 一致)
  collect_features:
    q_features: true                             # 收集 Query 特征
    k_features: true                             # 收集 Key 特征
    v_features: true                             # 收集 Value 特征
    out_features: true                           # 收集 Output 特征
  cross_attention_layers: "all"                  # "all" 或指定层索引列表
```

## 存储优化和注意事项

### 1. 存储空间估算

对于标准 Stable Diffusion 模型：
- Cross-attention 层数: ~16 层
- 特征类型: 4 种 (Q/K/V/OUT)
- 去噪步数: 50 步
- 特征维度: ~320-1280 (取决于层)

每个协方差矩阵大小: `feature_dim² × 4 bytes`
总存储空间估算: `16 × 4 × 50 × 1280² × 4 bytes ≈ 26 GB`

### 2. 内存优化策略

```python
# 分批处理，避免内存溢出
def process_in_batches(prompts, batch_size=10):
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # 处理批次
        yield batch

# 即时保存，释放内存
def save_immediately_and_clear(features, output_dir):
    # 计算并保存协方差矩阵
    compute_and_save_covariance(features, output_dir)
    # 清理特征缓存
    features.clear()
    torch.cuda.empty_cache()
```

### 3. 错误处理

```python
def robust_save_matrix(matrix, filepath):
    """安全保存矩阵文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存临时文件
        temp_path = filepath + ".tmp"
        np.save(temp_path, matrix)
        
        # 原子性重命名
        os.rename(temp_path, filepath)
        
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
```

## 验证和诊断

### 协方差矩阵质量检查

```python
def validate_covariance_matrix(cov_matrix):
    """验证协方差矩阵的数学性质"""
    
    # 检查对称性
    is_symmetric = np.allclose(cov_matrix, cov_matrix.T, rtol=1e-5)
    
    # 检查半正定性
    eigenvals = np.linalg.eigvals(cov_matrix)
    is_positive_semidefinite = np.all(eigenvals >= -1e-8)
    
    # 检查数值稳定性
    condition_number = np.linalg.cond(cov_matrix)
    is_well_conditioned = condition_number < 1e12
    
    return {
        'symmetric': is_symmetric,
        'positive_semidefinite': is_positive_semidefinite, 
        'well_conditioned': is_well_conditioned,
        'condition_number': condition_number,
        'rank': np.linalg.matrix_rank(cov_matrix)
    }
```

## 未来扩展

### 1. 增量更新

```python
def update_covariance_incrementally(old_cov, new_features, old_n, new_n):
    """增量更新协方差矩阵，避免重新计算全部数据"""
    total_n = old_n + new_n
    new_cov = compute_covariance(new_features)
    
    # 加权平均更新
    updated_cov = (old_n * old_cov + new_n * new_cov) / total_n
    return updated_cov
```

### 2. 压缩存储

```python
def compress_covariance_matrix(cov_matrix, compression_ratio=0.95):
    """基于 SVD 的协方差矩阵压缩"""
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # 保留主要奇异值
    k = int(len(S) * compression_ratio)
    U_compressed = U[:, :k]
    S_compressed = S[:k]
    Vt_compressed = Vt[:k, :]
    
    return U_compressed, S_compressed, Vt_compressed
```

### 3. 分布式计算

```python
def distribute_covariance_computation(prompts, num_workers=4):
    """分布式计算协方差矩阵"""
    from multiprocessing import Pool
    
    # 将 prompts 分割到多个进程
    chunks = np.array_split(prompts, num_workers)
    
    with Pool(num_workers) as pool:
        partial_results = pool.map(compute_partial_covariance, chunks)
    
    # 合并结果
    return merge_covariance_results(partial_results)
```

---

**注意**: 本文档描述的存储逻辑基于 AlphaEdit 论文的理论基础，并针对 Diffusion 模型的特殊性进行了适配。在实际使用中，请根据具体的模型架构和计算资源调整相关参数。