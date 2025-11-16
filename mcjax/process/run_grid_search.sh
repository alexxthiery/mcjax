#!/bin/bash
set -e

# Timestamp for results folder
timestamp=$(date +%Y%m%d_%H%M%S)
# timestamp='20251113_171725'

save_model=false

# Parameter grids
algos=('mcd' 'pis' 'dds')
targets=('1d' 'funnel')
loss_types=('kl' 'lv')
Ks=(200)
lrs=(0.00001 0.0001 0.001)
seeds=(0)

# Main grid search loop
for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for loss in "${loss_types[@]}"; do
      for K in "${Ks[@]}"; do
        for lr in "${lrs[@]}"; do
          for seed in "${seeds[@]}"; do
            echo ""
            echo "Running ${algo}-${target}-${loss}, K=${K}, lr=${lr}, seed=${seed}"
            python main.py \
              --K "$K" \
              --T 1 \
              --sigma 1.0 \
              --lr "$lr" \
              --batch_size 256 \
              --num_steps 5000 \
              --to_visualize false \
              --get_metrics true \
              --algo "$algo" \
              --target_dist "$target" \
              --if_logZ true \
              --if_train true \
              --save_model "$save_model" \
              --seed "$seed" \
              --set_timestamp true \
              --timestamp "$timestamp" \
              --model_path "models/${algo}_${target}_${loss}_${K}_${lr}_${seed}.pkl" \
              --use_control_in_denominator true \
              --samples_for_final_visualization 2000 \
              --loss_type "$loss" \
              --do_grid_search true || \
              echo "Failed: ${algo}-${target}-${loss} (K=${K}, lr=${lr}, seed=${seed})"
          done
        done
      done
    done
  done
done

# Summarize all results
echo ""
echo "Summarizing results into Excel..."
target_folder_path="results/$timestamp" 
python -c "from main import grid_summarize; grid_summarize('${target_folder_path}', '${target_folder_path}/metric_results_grid.xlsx')"

echo ""
echo "Grid search finished. Results saved to ${target_folder_path}"