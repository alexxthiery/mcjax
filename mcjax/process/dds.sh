#!/bin/bash
############################################
# ----------- Parameter settings -----------
############################################

# ----------- Network parameters -----------
if_train=true
model_path='model_params.pkl'
condition_term='grad_score' # 'grad_score': concatenate \nabla log p_target; 'score': concatenate log p_target; 'none': no condition term; 

# ---------- Target distribution parameters -----------
target_dist='gmm40' # 'gmm40': 40-component Gaussian Mixture Model; '1d': 1-d Gaussian Mixture Model


# ----------- Loss parameters -----------
if_animation=true # Generate animation of the backward process
add_score=true # Add score term to the loss function

# ----------- Process parameters -----------
variable_ts=false # Use variable time steps
K=2000 # Number of steps in the process
sigma=0.1 # Noise scale for the backward process

# ---------- Training parameters -----------
lr=0.001 # Learning rate
batch_size=128 # Batch size
num_steps=4000 # Number of steps for training
if_logZ=false # calculate logZ during training



python dds.py \
  --if_train $if_train \
  --model_path $model_path \
  --condition_term $condition_term \
  --target_dist $target_dist \
  --if_animation $if_animation \
  --add_score $add_score  \
  --variable_ts $variable_ts \
  --K $K \
  --sigma $sigma \
  --lr $lr \
  --batch_size $batch_size \
  --num_steps $num_steps \
  --if_logZ $if_logZ
