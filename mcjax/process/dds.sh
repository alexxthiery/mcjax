#!/bin/bash
############################################
# ----------- Parameter settings -----------
############################################

# ----------- Network parameters -----------
if_train=true
network_name='resblock' # 'mlp': Multi-Layer Perceptron;  'resblock': ResBlock model
condition_term='grad_score' # 'grad_score': concatenate \nabla log p_target; 'score': concatenate log p_target; 'none': no condition term; 

# ---------- Target distribution parameters -----------
target_dist='1d' # 'gmm40': 40-component Gaussian Mixture Model; '1d': 1-d Gaussian Mixture Model


# ----------- Loss parameters -----------
if_animation=true # Generate animation of the backward process
add_score=true # Add score term to the loss function

# ----------- Process parameters -----------
variable_ts=true # Use variable time steps
K=2000 # Number of steps in the process
sigma=1.0 # Noise scale for the backward process

# ---------- Training parameters -----------
lr=0.0001 # Learning rate
batch_size=128 # Batch size
num_steps=1000 # Number of steps for training
if_logZ=false # calculate logZ during training
model_path='model_params.pkl'



python dds.py \
  --if_train $if_train \
  --network_name $network_name \
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
