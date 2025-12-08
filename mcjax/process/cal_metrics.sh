#!/bin/bash
set -e

# timestamp='20251128_142529'
timestamp=$(date +%Y%m%d_%H%M%S)

#############################

algos=('mcd' 'dds' 'pis')
targets=('gmm40')
loss_types=('kl' 'lv')
seeds=(0 1 2)
dims=(2)
num_steps=3000

for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for loss_type in "${loss_types[@]}"; do
      for dim in "${dims[@]}"; do
        for seed in "${seeds[@]}"; do
          echo "Running $algo on $target DIM=$dim, loss_type=$loss_type, seed=$seed"
          python main.py \
            --K 200 \
            --T 1 \
            --sigma 1.0 \
            --lr 0.0001 \
            --batch_size 256 \
            --sigma 5.0 \
            --num_steps $num_steps \
            --to_visualize true \
            --get_metrics true \
            --algo $algo \
            --target_dist $target \
            --if_logZ true \
            --if_train true \
            --seed $seed \
            --set_timestamp true \
            --timestamp $timestamp \
            --model_path "models/${algo}_${target}_DIM=${dim}_${loss_type}_${seed}.pkl" \
            --use_control_in_denominator true \
            --samples_for_final_visualization 2000 \
            --loss_type $loss_type \
            --dw_draw_marginals true \
            --dw_draw_well_hist true
        done
      done
    done
  done
done



# summarize to Excel
target_folder_path="results/$timestamp" 
python -c "from main import summarize_results; summarize_results('$target_folder_path','$target_folder_path/metric_results.xlsx')"
