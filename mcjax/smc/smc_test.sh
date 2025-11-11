#!/bin/bash
set -e

num_steps=10 # Number of intermediate distributions (used only if we choose to use non-adaptive(preset) temperature
             # (if_adaptive=False)
num_runs=200 # Number of independent runs to estimate the normalizing constant
num_particles_arr=(100 500 1000 5000 10000) # Different number of particles to test
step_size=1.0 # Step size for the MCMC kernels
num_substeps=10 # Number of MCMC steps between two intermediate distributions
if_adaptive=True # Whether to use adaptive temperature scheme
reference='gaussian' # Reference distribution
target='funnel' # Target distribution
dim=2 # Dimensionality of the distributions
method='MALA' # MCMC method to use (choose from 'RWM' or 'MALA')

#============================================================================================   
######### Test 1: Run SMC with different number of particles and estimate the normalizing constant ##########
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
  --method "$method"