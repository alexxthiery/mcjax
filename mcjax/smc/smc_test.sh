#!/bin/bash
set -e

num_steps=10 # Number of intermediate distributions (used only if we choose to use non-adaptive(preset) temperature
             # (if_adaptive=False)
num_runs=200 # Number of independent runs to estimate the normalizing constant
num_particles_arr=(100 500 1000 5000 10000) # Different number of particles to test
num_particles=1000 # Number of particles (used only in Test 2)
step_size=0.5 # Step size for the MCMC kernels
num_substeps=10 # Number of MCMC steps between two intermediate distributions
max_step=500 # Maximum number of adaptive steps for MCMC kernels
ess_normalized_target=0.6 # Target normalized ESS for adaptive temperature scheme

if_adaptive=True # Whether to use adaptive temperature scheme
reference='gaussian' # Reference distribution
target='funnel' # Target distribution
dim=2 # Dimensionality of the distributions (used in targets with variable dimension)
method='MALA' # MCMC method to use (choose from 'RWM' or 'MALA')

alpha=0.2 # Blending factor for adaptive covariance in Geometric SMC
m=5 # Number of wells (only used in doublewell target)
delta=4.0 # distance between two wells (only used in doublewell target)

#============================================================================================   
######### Test 1: Run SMC with different number of particles and estimate the normalizing constant ##########
targets=("student" "mixedgaussian" "banana2d" "funnel")
methods=("RWM" "MALA")
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

target='doublewell'
dims=(5 10 20)
m=5 
delta=4.0 
for dim in "${dims[@]}"; do
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
######### Test 2: Run adapative/non-adaptive methods with 2 kernels on 2-component mixed gaussian in different dimensions
dimensions=(2 5 10 20 50)
target='mixedgaussian' 
for dim in "${dimensions[@]}"; do
  python smc_analysis.py \
    --num_steps "$num_steps" \
    --num_runs "$num_runs" \
    --num_particles "$num_particles" \
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