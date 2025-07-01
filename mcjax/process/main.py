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
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--inner_iters", type=int, default=100)
    parser.add_argument("--outer_iters", type=int, default=100)
    parser.add_argument("--num_samples_per_outer", type=int, default=1000)
    parser.add_argument("--use_control_in_denominator", type=str2bool, default=False)
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

    # Training or Load
    key = jr.PRNGKey(args.seed)
    if args.if_train:
        print(f"Start training with {args.algo}")
        key, sub = jr.split(key)
        t1 = time.time()
        final_state, final_key, losses, logz_vals, logz_vars,*buffer_info = alg.train(sub)
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
            # plot buffer data (hist) every 10 steps
            print("Plotting buffer data histograms...")
            buffer_data, buffer_size = buffer_info
            for i in range(0, len(buffer_data), 10):
                plt.figure()
                plt.hist(buffer_data[i,:buffer_size[i].astype(int),0], bins=50, density=True, alpha=0.5)
                plt.title(f"Buffer data at step {i}")
                plt.xlabel("x")
                plt.ylabel("Density")
                plt.savefig(f"{args.results_dir}/{args.algo}_buffer_step_{i}.png")
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

    # generate true score_seq for 1d mixed gaussian case
    if args.target_dist == "1d":
        print("score_seq shape:", score_seq.shape)
        true_score_seq = np.zeros((args.K,10000))
        for t in range(args.K):
            y = samples_seq[t, :, :]
            true_score_seq[t,:] = alg.ou.ou_mixture_score(y, args.K - t - 1,alg.target_dist.mu,\
                                                            alg.target_dist.sigma, jnp.exp(alg.target_dist.log_w))

        # plot the true_score_seq and network-generated score_seq (mean at each time step)
        plt.figure(figsize=(10, 5))
        plt.plot(true_score_seq.mean(axis=0), label="True Score", color='blue')
        plt.plot(score_seq.mean(axis=0), label="Estimated Score", color='orange')
        plt.xlabel("Sample Index")
        plt.ylabel("Score")
        plt.title("True vs Estimated Score at Final Time Step")
        plt.legend()
        plt.savefig(f"{args.results_dir}/{args.algo}_true_vs_estimated_score.png")
        plt.close()




    # Visualization 
    # alg.visualize_samples(samples_seq)



if __name__ == "__main__":
    print(f"Available devices: {jax.devices()}")
    jax.config.update("jax_platform_name", "gpu")
    main()
