#!/bin/bash

# Define datasets
datasets=("colas" "hall" "t1d" "T1DEXI_adults")

# Define kernels
kernels=("matern32" "matern52" "squared_exponential")

# Define percentiles
percentiles=(0.95 0.9 0.8 0.7)

# Run commands
for dataset in "${datasets[@]}"; do
    for kernel in "${kernels[@]}"; do
        for percentile in "${percentiles[@]}"; do
            echo "Running: python main.py --dataset $dataset --method gp --percentile $percentile --kernel $kernel"
            python main.py --dataset "$dataset" --method gp --percentile "$percentile" --kernel "$kernel"
        done
    done
done
