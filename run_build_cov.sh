#!/bin/bash

# NullBooth Covariance Matrix Builder
# 支持独立运行Phase 1或Phase 2，或两者都运行

# 默认运行both phases
PHASE="${1:-both}"

# 检查参数
if [[ "$PHASE" != "1" && "$PHASE" != "2" && "$PHASE" != "both" ]]; then
    echo "Usage: $0 [1|2|both]"
    echo "  1: Run only Phase 1 (Feature Collection)"
    echo "  2: Run only Phase 2 (Covariance Computation)"
    echo "  both: Run both phases (default)"
    exit 1
fi

echo "Starting NullBooth Covariance Matrix Builder - Phase: $PHASE"
echo "Timestamp: $(date)"

# 运行build_cov_parallel.py with phase参数
accelerate launch --config_file ./accelerate_config.yaml build_cov_parallel.py \
    --config configs/nullbooth-LCM.yaml \
    --phase $PHASE \
    --resume

echo "Completed at: $(date)"