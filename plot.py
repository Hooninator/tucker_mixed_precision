import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import hashlib
import os

from dataclasses import dataclass
from collections import defaultdict
from matplotlib.cm import get_cmap

path = "./results/results_v3/"
plotpath = "./plots/plots_v3/"


@dataclass(eq=True, frozen=True)
class Config:

    tensor: str
    lra: str
    qr: str
    init: str
    ttmc_u: str
    lra_u: str
    ranks: str

    def get_label(self):
        return f"{self.lra}-{self.qr}-lra_{self.lra_u}-ttmc_{self.ttmc_u}"

    def get_label_short(self):
        return f"svd_{self.lra_u}-ttmc_{self.ttmc_u}"

    def get_color(self):
        cmap_name = 'tab10'
        label = self.get_label()
        h = hashlib.sha256(label.encode('utf-8')).hexdigest()
        int_hash = int(h[:8], 16)  # take first 8 hex digits
        normalized = (int_hash % 1000) / 1000.0
        cmap = get_cmap(cmap_name)
        return cmap(normalized)


    
def parse_filename(fname):
    bits = fname.split(".csv")[0].split('-')
    args = []
    for bit in bits:
        args.append(bit.split(":")[-1])
    #args[0] = args[0].split("/")[-1]
    print(args)
    return Config(*args)


def parse_csvs(prefix):
    files = os.listdir(path)
    df_dict = {}
    for filename in files:
        if prefix not in filename:
            continue
        config = parse_filename(filename[len(prefix)+1:])
        filename = path + filename
        with open(filename, 'r') as file:
            df = pd.read_csv(filename)
            df = df.iloc[:, 1:]
            df_dict[config] = df
    return df_dict


def group_by_tensor_and_rank(df_dict):
    grouped = defaultdict(list)
    for config in df_dict:
        tensor, rank = config.tensor, config.ranks
        grouped[(tensor, rank)].append(config)
    return grouped


def plot_config(config, df):
    x_inds = np.arange(df.shape[1])
    means = df.mean(axis=0)
    variance = df.var(axis=0)
    plt.plot(x_inds, means, marker='x', markersize=10, label=config.get_label(), color=config.get_color())
    plt.fill_between(x_inds, means + variance, means - variance)



def plot_accuracy(df_dict):
    # (tensor, rank) => List[config], use to index df_dict
    grouped_data = group_by_tensor_and_rank(df_dict)
    for tensor_rank in grouped_data:
        for config in grouped_data[tensor_rank]:
            plot_config(config, df_dict[config])
        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        plt.xlabel("Iteration")
        plt.ylabel("Relative Reconstruction Error")
        plt.title(f"Reconstruction Error: {tensor_rank[0]} - {tensor_rank[1]}")
        plt.legend()
        plt.savefig(f"{plotpath}{tensor_rank[0]}_{tensor_rank[1]}", bbox_inches='tight')
        plt.clf()


def plot_timing(df_dict):
    grouped_data = group_by_tensor_and_rank(df_dict)
    for tensor_rank in grouped_data:
        for config in grouped_data[tensor_rank]:
            plot_config(config, df_dict[config].iloc[:, 1:])
        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        plt.xlabel("Iteration")
        plt.ylabel("Runtime (s)")
        plt.title(f"Runtime: {tensor_rank[0]} - {tensor_rank[1]}")
        plt.legend()
        plt.savefig(f"{plotpath}{tensor_rank[0]}_{tensor_rank[1]}_timing", bbox_inches='tight')
        plt.clf()


def plot_breakdown(df_dict):

    phases = ["ttmc", "factor_update", "scaling"]
    colors = {"ttmc":"crimson", "factor_update":"steelblue", "scaling":"salmon"}
    order = {"svd_fp64-ttmc_fp64":0, "svd_fp32-ttmc_fp64":1, "svd_fp32-ttmc_fp16":2}

    grouped_data = group_by_tensor_and_rank(df_dict)
    for tensor_rank in grouped_data:
        bottom = np.zeros(len(grouped_data[tensor_rank]))
        for phase in phases:
            y_arr = [0] * len(grouped_data[tensor_rank])
            i = 0

            for config in grouped_data[tensor_rank]:
                y_arr[order[config.get_label_short()]] = (df_dict[config][phase].mean())

            plt.bar(np.arange(len(grouped_data[tensor_rank])), y_arr, bottom=bottom, label=phase, color=colors[phase], edgecolor='black', linewidth=1, zorder=2)
            bottom += np.array(y_arr)

        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        plt.xticks(np.arange(len(grouped_data[tensor_rank])), labels=list(order.keys())[:len(grouped_data[tensor_rank])])
        plt.ylabel("Runtime (s)")
        plt.title(f"Runtime Breakdown: {tensor_rank[0]} - {tensor_rank[1]}")
        plt.legend()
        plt.savefig(f"{plotpath}{tensor_rank[0]}_{tensor_rank[1]}_breakdown", bbox_inches='tight')
        plt.clf()


def get_total(df):
    means = df.mean()
    return sum(means)


def plot_speedup(df_dict):
    grouped_data = group_by_tensor_and_rank(df_dict)
    order = {"svd_fp64-ttmc_fp64":0, "svd_fp32-ttmc_fp64":1, "svd_fp32-ttmc_fp16":2}
    for tensor_rank in grouped_data:
        n = len(grouped_data[tensor_rank])
        y_arr = []
        for config in grouped_data[tensor_rank]:
            label = config.get_label_short()
            if order[label] == 0:
                baseline = get_total(df_dict[config])
                break

        for config in grouped_data[tensor_rank]:
            label = config.get_label_short()
            if order[label] == 0:
                continue
            speedup = baseline / (get_total(df_dict[config]))
            y_arr.append(speedup)
        fig, ax = plt.subplots()
        bars = ax.bar(np.arange(len(y_arr)), y_arr, color="limegreen", edgecolor='black', linewidth=1, zorder=2)
        ax.bar_label(bars, fmt=lambda x: f'{x:.2f}x')

        plt.rcParams['font.size'] = 12

        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        plt.xticks(np.arange(len(y_arr)), labels=list(order.keys())[1:1+len(y_arr)])
        plt.ylabel("Speedup")
        plt.title(f"Speedup: {tensor_rank[0]} - {tensor_rank[1]}")
        plt.legend()
        plt.savefig(f"{plotpath}{tensor_rank[0]}_{tensor_rank[1]}_speedup", bbox_inches='tight')
        plt.clf()


if __name__=="__main__":
    timing_df_dict = parse_csvs("timing")
    #plot_breakdown(timing_df_dict)
    plot_speedup(timing_df_dict)
    err_df_dict = parse_csvs("error")
    plot_accuracy(err_df_dict)


