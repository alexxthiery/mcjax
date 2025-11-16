# main.py

import argparse
from ast import For
import re
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
import numpy as np
import pandas as pd
import glob, os
import torch
import json
from datetime import datetime

from algo import DDSAlgorithm,IDEMAlgorithm, PISAlgorithm, ControlledMonteCarloDiffusion
from metrics import MMD_squared,two_wasserstein,sinkhorn_distance

from mcjax.proba.doublewell import DoubleWell

from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1', 'yes')
    parser = argparse.ArgumentParser(description="Neural Sampler Experiments")
    parser.add_argument("--algo",       type=str, default="dds",
                        choices=["dds", "pis", "idem", "mcd", "cmcd"])
    parser.add_argument("--target_dist",  type=str, default="1d")
    parser.add_argument("--network_name", type=str, default="resblock",
                        choices=["mlp", "resblock"])
    parser.add_argument("--condition_term", type=str, default="grad_score",
                        choices=["none", "score", "grad_score"])
    parser.add_argument('--add_score', type=str2bool, default=True) 
    parser.add_argument('--variable_ts', type=str2bool, default=False)  
    parser.add_argument("--K",          type=int, default=200)
    parser.add_argument("--T",          type=int, default=1)  
    parser.add_argument("--sigma",      type=float, default=1.0)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_steps",  type=int, default=1000)
    parser.add_argument("--if_logZ",    type=str2bool, default=True)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--if_train",   type=str2bool, default=True)
    parser.add_argument("--save_model", type=str2bool, default=True)
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
    parser.add_argument("--use_control_in_denominator", type=str2bool, default=True)
    parser.add_argument("--to_visualize", type=str2bool, default=True)
    parser.add_argument("--get_metrics", type=str2bool, default=True)
    parser.add_argument("--write_logZ", type=str2bool, default=True)
    parser.add_argument("--set_timestamp", type=str2bool, default=False)
    parser.add_argument("--timestamp", type=str, default="default_timestamp")
    parser.add_argument("--samples_for_final_visualization", type=int, default=10000)
    parser.add_argument("--loss_type", type=str, default="kl",
                        choices=["kl", "lv"])
    parser.add_argument("--sde_ctrl_noise", type=float, default=0.0)
    parser.add_argument("--do_grid_search", type=str2bool, default=False)
    parser.add_argument("--dw_draw_marginals", type=str2bool, default=False)
    parser.add_argument("--dw_draw_well_hist", type=str2bool, default=False)
    return parser.parse_args()

def append_metrics(algo, target, loss_type, delta_logZ, wass, sink, target_folder_path):
    os.makedirs(target_folder_path, exist_ok=True)
    fname = os.path.join(target_folder_path, f"metrics_{algo}_{target}.json")

    if os.path.exists(fname):
        try:
            with open(fname, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}  # Handle empty or corrupted file
    else:
        data = {}

    # Ensure the loss_type key and its nested dict exist
    if loss_type not in data:
        data[loss_type] = {"delta_logZ": [], "wasserstein": [], "sinkhorn": []}
    
    # Ensure all metric keys exist 
    for metric in ["delta_logZ", "wasserstein", "sinkhorn"]:
        if metric not in data[loss_type]:
            data[loss_type][metric] = []

    data[loss_type]["delta_logZ"].append(float(delta_logZ))
    data[loss_type]["wasserstein"].append(float(wass))
    data[loss_type]["sinkhorn"].append(float(sink))

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Appended new metrics to {fname} under loss type '{loss_type}'")

def append_metrics_grid_search(algo, target, loss_type,
                               delta_logZ, wass, sink,
                               target_folder_path, K=None, lr=None):
    """
    Append metrics for one experiment run into JSON grouped by (algo, loss_type).
    Structure:
    {
        "target_name": {
            "K100_lr0.001": {
                "delta_logZ": [...],
                "wasserstein": [...],
                "sinkhorn": [...]
            },
            ...
        }
    }
    """
    import os, json

    os.makedirs(target_folder_path, exist_ok=True)
    fname = os.path.join(target_folder_path, f"metrics_{algo}_{loss_type}.json")

    if os.path.exists(fname):
        try:
            with open(fname, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    key_hyper = f"K{K}_lr{lr}"
    if target not in data:
        data[target] = {}
    if key_hyper not in data[target]:
        data[target][key_hyper] = {"delta_logZ": [], "wasserstein": [], "sinkhorn": []}

    for metric, val in [("delta_logZ", delta_logZ), ("wasserstein", wass), ("sinkhorn", sink)]:
        data[target][key_hyper][metric].append(float(val))

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Appended metrics for {algo}-{loss_type}-{target} ({key_hyper})")


def summarize_results(target_folder_path, excel_path="metric_results.xlsx"):
    """
    Summarize metrics from all JSON files under a single target folder and save to Excel.
    
    The function recursively searches for JSON files, extracts Target and Algorithm
    names from the path/filename, calculates mean metrics, and pivots the results.
    """
    records = []
    
    # Search recursively for all .json files
    search_path = os.path.join(target_folder_path, "**", "*.json")
    
    for json_file in glob.glob(search_path, recursive=True):
        print(f"Processing {json_file}...")

        target_name = os.path.basename(os.path.dirname(json_file)) 
        
        filename_prefix = os.path.splitext(os.path.basename(json_file))[0]
        
        try:
            parts = filename_prefix.split('_')
            algo_name = parts[1]
        except IndexError:
            algo_name = filename_prefix  

        with open(json_file, "r") as f:
            try:
                data_all_losses = json.load(f)
            except json.JSONDecodeError:
                print(f"  Skipping corrupted file: {json_file}")
                continue

        for loss_type, metrics in data_all_losses.items():
            
            for metric_name, values in metrics.items():
                if not values: 
                    mean_val = np.nan
                else:
                    mean_val = np.mean(values)

                records.append((target_name, algo_name, loss_type, metric_name, mean_val))


    if not records:
        print(f"No valid JSON data found in {target_folder_path} or its subdirectories.")
        return


    df = pd.DataFrame(records, columns=["Target", "Algorithm", "Loss", "Metric", "Mean Value"])


    loss_map = {
        'lv': 'Log-variance',
        'kl': 'Kullback-Leibler'
    }
    df['Loss'] = df['Loss'].map(loss_map).fillna(df['Loss'])

    pivot_df = df.pivot_table(
        index=["Target", "Loss", "Algorithm"], 
        columns="Metric", 
        values="Mean Value"
    )


    metric_order = ['delta_logZ', 'sinkhorn', 'wasserstein']

    existing_cols = [col for col in metric_order if col in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=existing_cols)

    pivot_df.index.names = ['Problem', 'Loss', 'Method']
    
    pivot_df = pivot_df.sort_index(level=['Problem', 'Loss', 'Method'])

    pivot_df.to_excel(excel_path)
    print(f"Summary written to {excel_path}")


def grid_summarize(target_folder_path, excel_path="metric_results_grid.xlsx"):
    """
    Summarize grid search results across multiple targets.
    """

    all_records = []

    pattern = os.path.join(target_folder_path, "**", "metrics_*.json")
    for json_file in glob.glob(pattern, recursive=True):
        base = os.path.basename(json_file)
        m = re.match(r"metrics_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)\.json", base)
        if not m:
            print(f"Skipping unexpected file: {json_file}")
            continue

        algo, loss = m.groups()

        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping corrupted file: {json_file}")
            continue
        for target, hp_dict in data.items():
            for hp_key, metrics in hp_dict.items():
                m2 = re.match(r"K(\d+)_lr([0-9.eE+-]+)", hp_key)
                if not m2:
                    print(f"Skipping unexpected hyperparam key: {hp_key}")
                    continue

                K_str, lr_str = m2.groups()
                K = int(K_str)
                lr = float(lr_str)

                for metric_name, vals in metrics.items():
                    if not vals:
                        continue
                    all_records.append({
                        "Method": algo,
                        "Loss": loss,
                        "Target": target,
                        "K": K,
                        "LR": lr,
                        "Metric": metric_name,
                        "Mean": np.mean(vals)
                    })

    df = pd.DataFrame(all_records)
    if df.empty:
        print("No valid metrics found.")
        return

    # Write results to Excel (one sheet per Method-Loss)
    writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    workbook = writer.book
    bold_fmt = workbook.add_format({"bold": True, "font_color": "green"})

    for (method, loss), sub_df in df.groupby(["Method", "Loss"]):
        pivot = sub_df.pivot_table(
            index=["Target", "K", "LR"],
            columns="Metric",
            values="Mean"
        )
        pivot_reset = pivot.reset_index()

        # Determine which metrics are actually present
        metric_cols = [c for c in ["delta_logZ", "wasserstein", "sinkhorn"]
                       if c in pivot_reset.columns]

        if not metric_cols:
            print(f"No metric columns found for {method}-{loss}, skipping.")
            continue
        '''
        Compute MeanRank PER TARGET
        For each Target separately, pick the best (K, LR).
        We rank all hyperparameter settings (K, LR) using .rank() on the metric columns.
        For each (K, LR) row, we average its rank across metrics 
        '''

        def compute_meanrank_per_target(group):
            ranks = group[metric_cols].rank()  # rank within this target
            return ranks.mean(axis=1)          

        meanrank_series = pivot_reset.groupby("Target", group_keys=False).apply(
            compute_meanrank_per_target
        )

        pivot_reset["MeanRank"] = meanrank_series

        best_idx_per_target = pivot_reset.groupby("Target")["MeanRank"].idxmin()

        # Mark which rows are "best" for their target
        pivot_reset["IsBest"] = pivot_reset.index.isin(best_idx_per_target)

        # Sort rows so that same targets are together in the sheet
        merged = pivot_reset.sort_values(["Target", "MeanRank", "K", "LR"]).reset_index(drop=True)

        sheet_name = f"{method}_{loss}"
        merged.drop(columns=["IsBest"]).to_excel(writer, sheet_name=sheet_name, index=False)

        # Boldify best row per target in the sheet
        worksheet = writer.sheets[sheet_name]
        for i, row in enumerate(merged.itertuples(), start=1): 
            if row.IsBest:
                worksheet.set_row(i, None, bold_fmt)

        print(f"\nBest hyperparameters for {method}-{loss}:")
        for target in merged["Target"].unique():
            best_rows = merged[(merged["Target"] == target) & (merged["IsBest"])]
            for _, r in best_rows.iterrows():
                print(f"Target={target}: K={r['K']}, lr={r['LR']}, MeanRank={r['MeanRank']:.3f}")

    writer.close()
    print(f"\nGrid results saved to {excel_path}")


def main():
    args = parse_args()
    # create results_dir if not exist
    os.makedirs(args.results_dir, exist_ok=True)

    if args.set_timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # create folder for this timestamp
    folder_path = os.path.join(args.results_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # add folder_path to args
    args.folder_path = folder_path

    # create models dir if not exist
    if args.save_model:
        os.makedirs("models", exist_ok=True)

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

    # create target_dist results dir if not exist
    target_folder_path = f"{args.folder_path}/{args.target_dist}"
    os.makedirs(target_folder_path, exist_ok=True)
    
    if args.algo == "idem" and args.backdiffusion_true_score and args.target_dist == "1d":
        print("Using true score function for backdiffusion in IDEM...")
        key = jr.PRNGKey(0)
        seq, score_seq = alg.sample_backward_true(key, num_samples=10000)
        figname = "backdiffusion_true_score"
        alg.visualize_samples(seq)
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
        if args.save_model:
            with open(args.model_path, "wb") as f:
                pickle.dump(final_state.params, f)
            print(f"Model parameters saved to {args.model_path}")

        # Plot loss curve
        plt.figure()
        plt.plot(losses, label="train loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"{args.algo} training loss K={args.K} steps={args.num_steps} loss type={args.loss_type}")
        plt.savefig(f"{args.folder_path}/{args.target_dist}/{args.algo}_{args.loss_type}_{args.K}_{args.lr}_loss.png")
        plt.close()

        if args.algo == "idem" and args.target_dist == "1d":
            # Delete old buffer histograms
            # search for files matching the pattern
            for filename in os.listdir(args.folder_path + '/' + args.target_dist):
                if filename.startswith(f"{args.algo}_buffer_step_") and filename.endswith(".png"):
                    os.remove(os.path.join(args.folder_path + '/' + args.target_dist, filename))

            # plot buffer data (hist) every 10 steps
            print("Plotting buffer data histograms...")
            buffer_data, buffer_size = buffer_info
            for i in range(1, len(buffer_data)+1, args.draw_buffer_interval):
                plt.figure()
                plt.hist(buffer_data[i-1,:buffer_size[i-1].astype(int),0], bins=50, density=True, alpha=0.5)
                plt.title(f"Buffer data at step {i}")
                plt.xlabel("x")
                plt.ylabel("Density")
                plt.savefig(f"{args.folder_path}/{args.target_dist}/{args.algo}_buffer_step_{i}.png")
                plt.close()
            # plot the final buffer data
            plt.figure()
            plt.hist(buffer_data[-1,:buffer_size[-1].astype(int),0], bins=50, density=True, alpha=0.5)
            plt.title(f"Final Buffer data at step {len(buffer_data)}")
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.savefig(f"{args.folder_path}/{args.target_dist}/{args.algo}_buffer_step_{len(buffer_data)}.png")
            plt.close()

            # plot diff_true_ests curve
            plt.figure()
            plt.plot(diff_true_ests, label="diff_true_est")
            plt.xlabel("step")
            plt.ylabel("diff_true_est")
            plt.legend()
            plt.title(f"{args.algo} diff_true_est")
            plt.savefig(f"{args.folder_path}/{args.target_dist}/{args.algo}_diff_true_est.png")
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

            # plot the horizontal line for true logZ
            ax2.axhline(y=alg.target_dist.log_Z(), color='C2', linestyle='--', label="true logZ")

            lines, labels = ax1.get_legend_handles_labels()
            l2, lbl2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + l2, labels + lbl2, loc='upper left')
            plt.title(f"{args.algo} logZ statistics K={args.K} steps={args.num_steps} loss type={args.loss_type}")
            plt.savefig(f"{args.folder_path}/{args.target_dist}/{args.algo}_{args.loss_type}_{args.K}_{args.lr}_logZ.png")
            plt.close()

            # output final logZ estimate
            logZ_est = float(logz_vals[len(x)-1])
            # true logZ from target distribution
            logZ_true = float(alg.target_dist.log_Z())
            delta_logZ = abs(logZ_est - logZ_true)

        
    else:
        # Load saved params and wrap into a dummy TrainState
        print(f"Loading model parameters from {args.model_path}")
        try:
            with open(args.model_path, "rb") as f:
                saved_params = pickle.load(f)
            alg.state = alg.state.replace(params=saved_params)
        except Exception as e:
            print(f"Failed to load model parameters: {e}")

    if args.to_visualize or args.get_metrics or args.dw_draw_marginals or args.dw_draw_well_hist:
        # Sampling
        key, sub = jr.split(key)
        samples_seq, score_seq = alg.sample(alg.state.params, sub, num_samples=args.samples_for_final_visualization)
        samples_seq = jax.device_get(samples_seq)  # shape (K, N, dim)
        score_seq = jax.device_get(score_seq)    
        

        # Visualization 
        if args.to_visualize:
            alg.visualize_samples(samples_seq)


        # Compute Metrics if target_dist can be sampled from (check if sample() method is implemented)
        if alg.target_dist.can_sample:
            final_samples = samples_seq[-1]
            tgt_samps = alg.target_dist.sample(jr.PRNGKey(999), args.samples_for_final_visualization)
            wass = two_wasserstein(np.array(final_samples), np.array(tgt_samps)) # wasserstein distance
            print(f"Wasserstein distance (p=2): {wass:.4e}")
            sink = sinkhorn_distance(
                torch.tensor(final_samples).clone(), 
                torch.tensor(np.array(tgt_samps)).clone(),
                max_iters=2000,
                eps=1e-3
            ).item() # sinkhorn distance
            print(f"Sinkhorn distance : {sink:.4e}")
        else:
            wass = float('nan')
            sink = float('nan')
            print("Target distribution cannot be sampled from, skipping metric computations.")

       
        # Append metrics to JSON
        if args.do_grid_search:
            append_metrics_grid_search(
                args.algo, args.target_dist, args.loss_type, 
                delta_logZ if args.if_logZ else float('nan'), 
                wass, sink, target_folder_path, args.K, args.lr
            )
        else:
            append_metrics(args.algo, args.target_dist, args.loss_type, delta_logZ if args.if_logZ else float('nan'), wass, sink, target_folder_path)

        if args.dw_draw_marginals and args.target_dist == "doublewell":
            print("Drawing DoubleWell marginals...")
            # doublewell = DoubleWell()
            print("delta:", alg.target_dist.delta, "m:", alg.target_dist.m)
            alg.target_dist.plot_doublewell_marginals(args.folder_path, args.algo, np.array(samples_seq[-1]), alg.target_dist.delta, alg.target_dist.m)

        if args.dw_draw_well_hist and args.target_dist == "doublewell":
            print("Drawing DoubleWell well histogram...")
            # doublewell = DoubleWell()
            alg.target_dist.plot_doublewell_well_hist(args.folder_path, args.algo, np.array(samples_seq[-1]), alg.target_dist.delta, alg.target_dist.m)
if __name__ == "__main__":
    print(f"Available devices: {jax.devices()}")
    jax.config.update("jax_platform_name", "gpu")
    main()
