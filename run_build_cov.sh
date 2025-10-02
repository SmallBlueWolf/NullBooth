#!/bin/bash
# Configure to use GPUs 4-7 (4x RTX 4090)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# NCCL environment configuration for better compatibility with 4 GPUs
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_TREE_THRESHOLD=0

# Additional optimizations for multi-GPU scaling
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

# Read sampler strategy from config file
SAMPLER_STRATEGY=$(python -c "import yaml; config = yaml.safe_load(open('configs/nullbooth.yaml')); print(config['nullbooth'].get('sampler_strategy', 'uniform'))" 2>/dev/null || echo "unknown")
NUM_STEPS=$(python -c "import yaml; config = yaml.safe_load(open('configs/nullbooth.yaml')); print(config['nullbooth'].get('num_denoising_steps', 30))" 2>/dev/null || echo "30")

# Run the covariance building with optimized configuration
echo "Starting covariance matrix building on GPUs 4-7 (4x RTX 4090)"
echo "Configuration:"
echo "  - Sampler strategy: $SAMPLER_STRATEGY"
echo "  - Number of denoising steps: $NUM_STEPS"
echo "  - Denoising: Always from max noise (999) to target timestep"
echo "  - Nullspace threshold: 1e-4 (or 10th percentile)"
echo ""

accelerate launch --config_file ./accelerate_config_4gpu.yaml build_cov_parallel.py --config configs/nullbooth.yaml