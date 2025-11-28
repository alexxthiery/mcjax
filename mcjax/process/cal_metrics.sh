#!/bin/bash
set -e

timestamp='20251128_142529'
# timestamp=$(date +%Y%m%d_%H%M%S)

#############################
# Compare the condition_term setting's influence on funnel target

algos=('mcd' 'dds')
targets=('diaggauss')
loss_types=('kl' 'lv')
dim=100
seeds=(0 1 2)

num_steps=3000
delta=4.0
m=5
offset=(0.0 1.0 -1.0 2.0 -2.0) 
dims=(5 10 20 50)

for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for loss_type in "${loss_types[@]}"; do
      for dim in "${dims[@]}"; do
        diaggauss_var=($(seq 1 "$dim")) 
        for seed in "${seeds[@]}"; do
          echo "Running $algo on $target DIM=$dim, loss_type=$loss_type, seed=$seed"
          python main.py \
            --K 200 \
            --T 1 \
            --sigma 1.0 \
            --lr 0.0001 \
            --batch_size 256 \
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
            --dw_draw_well_hist true \
            --dim $dim \
            --delta $delta \
            --m $m \
            --offset ${offset[@]} \
            --diaggauss_var ${diaggauss_var[@]}
        done
      done
    done
  done
done



# summarize to Excel
target_folder_path="results/$timestamp" 
python -c "from main import summarize_results; summarize_results('$target_folder_path','$target_folder_path/metric_results.xlsx')"
