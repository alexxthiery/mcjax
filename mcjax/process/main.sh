#!/bin/bash
############################################
# ----------- Parameter settings -----------
############################################

algo='idem' 
target_dist='1d' # 'gmm40': 40-component Gaussian Mixture Model; '1d': 1-d Gaussian Mixture Model
network_name='resblock' # 'mlp': Multi-Layer Perceptron;  'resblock': ResBlock model
condition_term='grad_score' # 'grad_score': concatenate \nabla log p_target; 'score': concatenate log p_target; 'none': no condition term;
add_score=true # Add score term to the loss function
variable_ts=true # Use variable time steps
K=2000 # Number of steps in the process
sigma=1.0 # Noise scale for the backward process
lr=0.0001 # Learning rate
batch_size=128 # Batch size
num_steps=1000 # Number of steps for training
if_logZ=true # calculate logZ during training
seed=42 # Random seed for reproducibility
if_train=true # Whether to train the model
if_animation=true # Generate animation of the backward process
model_path='model_params.pkl' # Path to save the model parameters
results_dir='results' # Path to save the results

buffer_size=5000 # Buffer size for the training data in IDEM algorithm
inner_iters=200 # Number of inner steps for the IDEM algorithm
outer_iters=20 # Number of outer steps for the IDEM algorithm
num_samples_per_outer=1000 # Number of samples per outer step for the IDEM algorithm


python main.py \
    --algo $algo \
    --target_dist $target_dist \
    --network_name $network_name \
    --condition_term $condition_term \
    --add_score $add_score \
    --variable_ts $variable_ts \
    --K $K \
    --sigma $sigma \
    --lr $lr \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --if_logZ $if_logZ \
    --seed $seed \
    --if_train $if_train \
    --if_animation $if_animation \
    --model_path $model_path \
    --results_dir $results_dir \
    --buffer_size $buffer_size \
    --inner_iters $inner_iters \
    --outer_iters $outer_iters \
    --num_samples_per_outer $num_samples_per_outer
  