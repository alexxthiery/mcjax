#!/bin/bash
set -e

timestamp='20251027_012728'
# timestamp=$(date +%Y%m%d_%H%M%S)

# algos=('dds')        
# targets=('gmmfixed' 'funnel' 'doublewell' 'doublewell2') 
# seeds=(0 1 2)

# for algo in "${algos[@]}"; do
#   for target in "${targets[@]}"; do
#     for seed in "${seeds[@]}"; do
#       echo "Running $algo on $target, seed=$seed"
#       python main.py \
#         --K 200 \
#         --T 1 \
#         --sigma 1.0 \
#         --lr 0.00001 \
#         --batch_size 1000 \
#         --num_steps 3000 \
#         --visualize_and_metrics true \
#         --algo $algo \
#         --target_dist $target \
#         --if_logZ true \
#         --if_train true \
#         --seed $seed \
#         --set_timestamp true \
#         --timestamp $timestamp \
#         --model_path "models/${algo}_${target}_${seed}.pkl" \
#         --use_control_in_denominator true 
#     done
#   done
# done

algos=('mcd')        
targets=('doublewell2') 
seeds=(0 1 2)

for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running $algo on $target, seed=$seed"
      python main.py \
        --K 200 \
        --T 1 \
        --sigma 1.0 \
        --lr 0.00001 \
        --batch_size 400 \
        --num_steps 3000 \
        --visualize_and_metrics true \
        --algo $algo \
        --target_dist $target \
        --if_logZ true \
        --if_train true \
        --seed $seed \
        --set_timestamp true \
        --timestamp $timestamp \
        --model_path "models/${algo}_${target}_${seed}.pkl" \
        --use_control_in_denominator true \
        --samples_for_final_visualization 10000 
    done
  done
done

# summarize to Excel
target_folder_path="results/$timestamp" 
python -c "from main import summarize_results; summarize_results('$target_folder_path','metric_results.xlsx')"
