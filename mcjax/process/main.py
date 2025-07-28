# main.py

import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('../../')
import time
from scipy.stats import gaussian_kde

from algo import DDSAlgorithm,IDEMAlgorithm, PISAlgorithm, ControlledMonteCarloDiffusion
from metrics import MMD_squared,two_wasserstein

from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1', 'yes')
    parser = argparse.ArgumentParser(description="Neural Sampler Experiments")
    parser.add_argument("--algo",       type=str, default="dds",
                        choices=["dds", "pis", "idem", "mcd", "cmcd"])
    parser.add_argument("--target_dist",     type=str, default="gmm40")
    parser.add_argument("--network_name", type=str, default="mlp",
                        choices=["mlp", "resblock"])
    parser.add_argument("--condition_term", type=str, default="grad_score",
                        choices=["none", "score", "grad_score"])
    parser.add_argument('--add_score', type=str2bool, default=False) 
    parser.add_argument('--variable_ts', type=str2bool, default=False)  
    parser.add_argument("--K",          type=int, default=2000)
    parser.add_argument("--T",          type=int, default=1)  
    parser.add_argument("--sigma",      type=float, default=1.0)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps",  type=int, default=1000)
    parser.add_argument("--if_logZ",    type=str2bool, default=False)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--if_train",   type=str2bool, default=False)
    parser.add_argument("--if_animation", type=str2bool, default=True)
    parser.add_argument("--model_path", type=str, default="model_params.pkl")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--sigma_min",  type=float, default=0.5)
    parser.add_argument("--sigma_max",  type=float, default=1.0)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--inner_iters", type=int, default=100)
    parser.add_argument("--outer_iters", type=int, default=100)
    parser.add_argument("--num_samples_per_outer", type=int, default=1000)
    parser.add_argument("--draw_buffer_interval", type=int, default=10)
    parser.add_argument("--num_samples_for_sk", type=int, default=10000)
    parser.add_argument("--debug_fill_buffer", type=str2bool, default=False)
    parser.add_argument("--backdiffusion_true_score", type=str2bool, default=False)
    parser.add_argument("--add_drift", type=str2bool, default=True)
    parser.add_argument("--sample_t_weight", type=str2bool, default=True)
    parser.add_argument("--use_control_in_denominator", type=str2bool, default=False)
    parser.add_argument("--use_true_score", type=str2bool, default=False)
    parser.add_argument("--visualize_forward", type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    # create results_dir if not exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Choose algorithm class
    if args.algo == "dds":
        AlgoClass = DDSAlgorithm
    elif args.algo == "pis":
        AlgoClass = PISAlgorithm
    elif args.algo == "idem":
        AlgoClass = IDEMAlgorithm
    elif args.algo == "mcd":
        AlgoClass = ControlledMonteCarloDiffusion
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not supported yet.")

    alg = AlgoClass(config=args)

    if args.visualize_forward and args.target_dist == "1d":
        # Visualize the forward process from mixed Gaussian to approx standard Gaussian
        print("Visualizing forward process...")
        key = jr.PRNGKey(0)
        alg.visualize_forward(key, num_samples=10000)
        return
    
    if args.algo == "idem" and args.backdiffusion_true_score and args.target_dist == "1d":
        print("Using true score function for backdiffusion in IDEM...")
        key = jr.PRNGKey(0)
        seq, score_seq = alg.sample_backward_true(key, num_samples=10000)
        figname = "backdiffusion_true_score"
        alg.visualize_samples(seq, figname=figname)
        return

    # Training or Load
    key = jr.PRNGKey(args.seed)
    if args.if_train:
        print(f"Start training with {args.algo}")
        key, sub = jr.split(key)
        t1 = time.time()
        if args.algo == "idem":
            final_state, final_key, losses, diff_true_ests, logz_vals, logz_vars, *buffer_info = alg.train(sub)
        else:
            final_state, final_key, losses, logz_vals, logz_vars = alg.train(sub)
        t2 = time.time()
        print(f"Tracing+Training finished in {t2 - t1:.2f} seconds.")
        alg.state = final_state  # Update the state with final trained parameters
        # Save parameters
        with open(args.model_path, "wb") as f:
            pickle.dump(final_state.params, f)

        # Plot loss curve
        plt.figure()
        plt.plot(losses, label="train loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"{args.algo} training loss")
        plt.savefig(f"{args.results_dir}/{args.algo}_loss.png")
        plt.close()

        if args.algo == "idem" and args.target_dist == "1d":
            # Delete old buffer histograms
            # search for files matching the pattern
            for filename in os.listdir(args.results_dir):
                if filename.startswith(f"{args.algo}_buffer_step_") and filename.endswith(".png"):
                    os.remove(os.path.join(args.results_dir, filename))

            # plot buffer data (hist) every 10 steps
            print("Plotting buffer data histograms...")
            buffer_data, buffer_size = buffer_info
            for i in range(1, len(buffer_data)+1, args.draw_buffer_interval):
                plt.figure()
                plt.hist(buffer_data[i-1,:buffer_size[i-1].astype(int),0], bins=50, density=True, alpha=0.5)
                plt.title(f"Buffer data at step {i}")
                plt.xlabel("x")
                plt.ylabel("Density")
                plt.savefig(f"{args.results_dir}/{args.algo}_buffer_step_{i}.png")
                plt.close()
            # plot the final buffer data
            plt.figure()
            plt.hist(buffer_data[-1,:buffer_size[-1].astype(int),0], bins=50, density=True, alpha=0.5)
            plt.title(f"Final Buffer data at step {len(buffer_data)}")
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.savefig(f"{args.results_dir}/{args.algo}_buffer_step_{len(buffer_data)}.png")
            plt.close()

            # plot diff_true_ests curve
            plt.figure()
            plt.plot(diff_true_ests, label="diff_true_est")
            plt.xlabel("step")
            plt.ylabel("diff_true_est")
            plt.legend()
            plt.title(f"{args.algo} diff_true_est")
            plt.savefig(f"{args.results_dir}/{args.algo}_diff_true_est.png")
            plt.close()

        # Plot logZ (if computed)
        if args.if_logZ:
            print("Plotting logZ statistics...")
            fig, ax1 = plt.subplots()
            x = 10 + jax.numpy.arange(args.num_steps // 10)*10 if args.algo != "idem" else args.inner_iters + jax.numpy.arange(args.outer_iters) * args.inner_iters
            ax1.plot(x, logz_vars[:len(x)], color='C0', label="logZ var")
            ax1.set_xlabel("step")
            ax1.set_ylabel("var(logZ)", color='C0')
            ax1.tick_params(axis='y', labelcolor='C0')

            ax2 = ax1.twinx()
            ax2.plot(x, logz_vals[:len(x)], color='C1', label="logZ mean")
            ax2.set_ylabel("mean(logZ)", color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')

            lines, labels = ax1.get_legend_handles_labels()
            l2, lbl2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + l2, labels + lbl2, loc='upper left')
            plt.title(f"{args.algo} logZ statistics")
            plt.savefig(f"{args.results_dir}/{args.algo}_logZ.png")
            plt.close()
        
        
    else:
        # Load saved params and wrap into a dummy TrainState
        print(f"Loading model parameters from {args.model_path}")
        with open(args.model_path, "rb") as f:
            saved_params = pickle.load(f)
        alg.state = alg.state.replace(params=saved_params)


    # Sampling
    key, sub = jr.split(key)
    samples_seq, score_seq = alg.sample(alg.state.params, sub, num_samples=10000)
    samples_seq = jax.device_get(samples_seq)  # shape (K, N, dim)
    score_seq = jax.device_get(score_seq)    
    

    # Metrics
    final_samples = samples_seq[-1]
    # Compute MMD between final_samples and target samples
    tgt_samps = alg.target_dist.sample(jr.PRNGKey(999), 10000)
    result = two_wasserstein(np.array(final_samples), np.array(tgt_samps))
    print(f"Wasserstein distance: {result:.4e}")

    # Visualization 
    if args.target_dist in ["1d", "gmm40", 'funnel','gmmfixed']:
        print("Visualizing samples...")
        if args.use_true_score and args.target_dist == "1d":
            figname = "true_score"
            alg.visualize_samples(samples_seq,figname=figname)
        else:
            figname = "estimated_score"
            alg.visualize_samples(samples_seq, figname=figname)


    # Plot -loss that should converge to ELBO
    if args.target_dist in ['sonar']:
        final_neg_loss = []
        for K in [10,50,100,200,500]:
            alg.cfg.K = K   
            print(f"Computing -loss estimate with K={K}...")    
            key, sub = jr.split(key)
            final_state, final_key, losses, logz_vals, logz_vars = alg.train(sub)
            final_neg_loss.append(-losses[-1])
        
        # print -loss against K
        plt.figure()
        plt.plot([10,50,100,200,500], final_neg_loss, marker='o')
        plt.xscale('log')
        plt.xlabel("K (number of steps)")
        plt.ylabel("-loss estimate")
        plt.title(f"{args.algo} -loss vs K")
        plt.savefig(f"{args.results_dir}/{args.target}_{args.algo}_loss_vs_K.png")
        plt.close()
       





if __name__ == "__main__":
    print(f"Available devices: {jax.devices()}")
    jax.config.update("jax_platform_name", "gpu")
    main()
