#!/bin/bash
set -e

algos=('mcd')        
targets=('gmmfixed' 'funnel' 'doublewell') 
seeds=(0 1 2 3 4)

for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running $algo on $target, seed=$seed"
      python main.py \
        --K 200 \
        --T 1 \
        --sigma 0.3 \
        --lr 0.00001 \
        --batch_size 1000 \
        --num_steps 10 \
        --do_visualization false \
        --algo $algo \
        --target_dist $target \
        --if_logZ true \
        --if_train true \
        --seed $seed \
        --results_dir results \
        --model_path "models/${algo}_${target}_${seed}.pkl" \
        --use_control_in_denominator true 
    done
  done
done

algos=('pis' 'dds')        
targets=('gmmfixed' 'funnel' 'doublewell') 
seeds=(0 1 2 3 4)

for algo in "${algos[@]}"; do
  for target in "${targets[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running $algo on $target, seed=$seed"
      python main.py \
        --K 200 \
        --T 1 \
        --sigma 1.0 \
        --lr 0.00001 \
        --batch_size 1000 \
        --num_steps 10 \
        --do_visualization false \
        --algo $algo \
        --target_dist $target \
        --if_logZ true \
        --if_train true \
        --seed $seed \
        --results_dir results \
        --model_path "models/${algo}_${target}_${seed}.pkl" \
        --use_control_in_denominator true 
    done
  done
done

# summarize to Excel
python -c "from main import summarize_results; summarize_results('results','metric_results.xlsx')"
