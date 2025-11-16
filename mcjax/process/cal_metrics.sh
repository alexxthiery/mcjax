#!/bin/bash
set -e

timestamp='20251108_162530'
# timestamp=$(date +%Y%m%d_%H%M%S)

algos=('pis' 'mcd' 'dds')  
targets=('funnel' '1d' 'gmmfixed' 'doublewell' 'pines')
loss_types=('kl' 'lv')
seeds=(0 1 2)

for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for loss_type in "${loss_types[@]}"; do
      for seed in "${seeds[@]}"; do
        echo "Running $algo on $target, loss_type=$loss_type, seed=$seed"
        python main.py \
          --K 200 \
          --T 1 \
          --sigma 1.0 \
          --lr 0.0001 \
          --batch_size 512 \
          --num_steps 3000 \
          --to_visualize true \
          --get_metrics true \
          --algo $algo \
          --target_dist $target \
          --if_logZ true \
          --if_train true \
          --seed $seed \
          --set_timestamp true \
          --timestamp $timestamp \
          --model_path "models/${algo}_${target}_${loss_type}_${seed}.pkl" \
          --use_control_in_denominator true \
          --samples_for_final_visualization 10000 \
          --loss_type $loss_type
      done
    done
  done
done


# summarize to Excel
target_folder_path="results/$timestamp" 
python -c "from main import summarize_results; summarize_results('$target_folder_path','metric_results.xlsx')"
