#!/bin/bash
set -e

num_steps=10 # Number of intermediate distributions (used only if we choose to use non-adaptive(preset) temperature
             # (if_adaptive=False)
num_runs=200 # Number of independent runs to estimate the normalizing constant
num_particles_arr=(100 500 1000 5000 10000) # Different number of particles to test
step_size=0.5 # Step size for the MCMC kernels
num_substeps=100 # Number of MCMC steps between two intermediate distributions
max_step=500 # Maximum number of adaptive steps for MCMC kernels
ess_normalized_target=0.6 # Target normalized ESS for adaptive temperature scheme

if_adaptive=True # Whether to use adaptive temperature scheme
reference='gaussian' # Reference distribution
target='funnel' # Target distribution
dim=2 # Dimensionality of the distributions (used in targets with variable dimension)
method='MALA' # MCMC method to use (choose from 'RWM' or 'MALA')

alpha=0.5 # Blending factor for adaptive covariance in Geometric SMC
m=5 # Number of wells (only used in doublewell target)
delta=4.0 # distance between two wells (only used in doublewell target)
sigma_x=3.0 # Standard deviation of the top-level variable in Neal's funnel (only used in funnel target)


#============================================================================================   
######### Test : Run SMC with different number of particles and estimate the normalizing constant ##########

dimensions=(2 5 10 20 50 100)
target='funnel' 
methods=('RWM' 'MALA')
for dim in "${dimensions[@]}"; do
    for method in "${methods[@]}"; do
      python smc_analysis.py \
      --num_steps "$num_steps" \
      --num_runs "$num_runs" \
      --num_particles_arr "${num_particles_arr[@]}" \
      --step_size "$step_size" \
      --num_substeps "$num_substeps" \
      --if_adaptive "$if_adaptive" \
      --reference "$reference" \
      --target "$target" \
      --dim "$dim" \
      --method "$method" \
      --alpha "$alpha" \
      --m "$m" \
      --delta "$delta"
  done
done


#============================================================================================   
######### Test : Get statistics for all doublewell, funnel, mixedgaussian targets under different dimensions ##########
python smc_analysis.py \
  --num_runs "$num_runs" \
  --step_size "$step_size" \
  --num_substeps "$num_substeps" \
  --reference "$reference" \
  --dim "$dim" \
  --method "$method" \
  --alpha "$alpha" \
  --m "$m" \
  --delta "$delta" \
  --sigma_x "$sigma_x"
