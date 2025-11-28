#!/bin/bash
############################################
# ----------- Parameter settings -----------
############################################

algo='mcd' # 'dds'： Denoising Diffusion Sampling; 
            # 'pis':path integral sampler; 
            #'idem': Iterative Denoising Estimation Method; 
            # 'mcd': Monte Carlo Denoising(Set use_control_in_denominator to true for CMCD, false for MCD)
target_dist='diaggauss' # 'gmm40': 40-component Gaussian Mixture Model; 
                # 'gmmfixed': Fixed 2-d gaussian with 9 components;
                # '1d': 1-d Gaussian Mixture Model； 
                # 'funnel': 2-d Funnel distribution;
                # 'doublewell': Double Well potential distribution;
                # 'pines': Log Gaussian Pines distribution;
                # 'diaggauss': Diagonal Gaussian distribution with self-defined diagonal variance;
network_name='resblock' # 'mlp': Multi-Layer Perceptron;  'resblock': ResBlock model
condition_term='grad_score' # 'grad_score': concatenate \nabla log p_target; 'score': concatenate log p_target; 'none': no condition term;
add_score=true # Add score term to the loss function
variable_ts=false # Use variable time steps; Always set this to false: No need to use variable time steps 
K=200 # Number of steps in the process 
T=1 # Time 
sigma=1.0 # Noise scale for the backward process
lr=1e-4 # Learning rate
batch_size=256 # Batch size
num_steps=3000 # Number of steps for training
if_logZ=true # calculate logZ during training
save_model=false # Whether to save the trained model parameters
seed=0 # Random seed for reproducibility
if_train=true # Whether to train the model
model_path='model_params.pkl' # Path to save the model parameters
results_dir='results' # Path to save the results
to_visualize=true # Whether to do visualization at the end of training
get_metrics=true # Whether to calculate metrics at the end of training
write_logZ=false # Whether to write logZ to txt file
set_timestamp=false # Whether to set a custom timestamp for saving results
samples_for_final_visualization=5000 # Number of samples for final visualization
loss_type='kl' # 'kl': KL divergence loss; 'lv': Least variance loss
sde_ctrl_noise=0.0 # float, stddev of Gaussian noise added to the detached SDE control (path) for exploration. Only used if loss_type='lv'.
do_grid_search=false # Whether this is called to do grid search of hyperparameters
dw_draw_marginals=true # Whether to draw DoubleWell marginals (1D and 2D) at the end of training
dw_draw_well_hist=true # Whether to draw DoubleWell well histogram at the end of training

############################################
# ----------- Parameters of targets -----------
dim=100 # Dimension of the target distribution (only used for targets with variable dimension, i.e: doublewell, funnel)
m=5 # Number of wells in DoubleWell target
delta=4.0 # Distance between wells in DoubleWell target
sigma_x=3.0 # noise standard deviation for Funnel target
offset=(0.0 1.0 -1.0 2.0 -2.0) # offset for each dimension in DoubleWell target
diaggauss_var=($(seq 1 "$dim"))  # diagonal variance for Diagonal Gaussian target



############################################
# ----------- For IDEM -----------
sigma_min=1.0 # Minimum noise scale for the IDEM algorithm
sigma_max=1.0 # Maximum noise scale for the IDEM algorithm
buffer_size=2000 # Buffer size for the training data in IDEM algorithm
inner_iters=1000 # Number of inner steps for the IDEM algorithm
outer_iters=5 # Number of outer steps for the IDEM algorithm
num_samples_per_outer=1000 # Number of samples per outer step for the IDEM algorithm
draw_buffer_interval=10 # Interval for drawing buffer histograms
num_samples_for_sk=10000 # Number of samples for estimating S_K in IDEM algorithm
debug_fill_buffer=true # Fill the buffer with samples from the target distribution for debugging

############################################
# ----------- For MCD -----------
use_control_in_denominator=true # True for CMCD, False for MCD

############################################'
# ------------ For debugging ---------------


python main.py \
    --algo $algo \
    --target_dist $target_dist \
    --network_name $network_name \
    --condition_term $condition_term \
    --add_score $add_score \
    --variable_ts $variable_ts \
    --K $K \
    --T $T \
    --sigma $sigma \
    --lr $lr \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --if_logZ $if_logZ \
    --save_model $save_model \
    --seed $seed \
    --if_train $if_train \
    --model_path $model_path \
    --results_dir $results_dir \
    --sigma_min $sigma_min \
    --sigma_max $sigma_max \
    --buffer_size $buffer_size \
    --inner_iters $inner_iters \
    --outer_iters $outer_iters \
    --num_samples_per_outer $num_samples_per_outer \
    --draw_buffer_interval $draw_buffer_interval \
    --num_samples_for_sk $num_samples_for_sk \
    --debug_fill_buffer $debug_fill_buffer \
    --use_control_in_denominator $use_control_in_denominator \
    --to_visualize $to_visualize \
    --get_metrics $get_metrics \
    --write_logZ $write_logZ \
    --set_timestamp $set_timestamp \
    --samples_for_final_visualization $samples_for_final_visualization \
    --loss_type $loss_type \
    --sde_ctrl_noise $sde_ctrl_noise \
    --do_grid_search $do_grid_search \
    --dw_draw_marginals $dw_draw_marginals \
    --dw_draw_well_hist $dw_draw_well_hist \
    --dim $dim \
    --m $m \
    --delta $delta \
    --sigma_x $sigma_x \
    --offset ${offset[@]} \
    --diaggauss_var ${diaggauss_var[@]}