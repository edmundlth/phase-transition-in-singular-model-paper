#!/bin/bash
OUTDIRNAME="./outputs/test_$(date +"%Y%m%d_%H%M")/"

python haiku_mlp_rlct_estimate.py --num-itemps 6 --num-posterior-samples 2000  --thinning 4 --num-warmup 1000 --num-chains 1 --num-training-data 2023 --a0 0.5 --b0 0.5 --sigma-obs 0.1 --prior-std 1.0 --prior-mean 0.0 --output_dir $OUTDIRNAME --rng-seed 42