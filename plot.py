import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import hashlib
import os

from dataclasses import dataclass
from collections import defaultdict
from matplotlib.cm import get_cmap

path = "./results/"
plotpath = "./plots/"


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
    args[0] = args[0].split("/")[-1]
    print(args)
    return Config(*args)


def parse_csvs():
    files = os.listdir(path)
    df_dict = {}
    for filename in files:
        filename = path + filename
        config = parse_filename(filename)
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


if __name__=="__main__":
    df_dict = parse_csvs()
    plot_accuracy(df_dict)


