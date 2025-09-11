#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file ./accelerate_config.yaml build_cov_parallel.py --config configs/nullbooth.yaml