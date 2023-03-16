# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import poptorch
import seaborn as sns
import torch
from torch_scatter import scatter


class ScatterOp(torch.nn.Module):

    def __init__(self,
                 num_inputs: int,
                 num_features: int,
                 num_outputs: int,
                 reduce: str = 'sum') -> None:
        super().__init__()
        self.scatter = partial(scatter,
                               dim_size=num_outputs,
                               reduce=reduce)
        input = torch.randn(num_inputs, num_features)
        index = torch.randint(num_outputs, (num_inputs, ))
        self.register_buffer('input', input)
        self.register_buffer('index', index)
        self.register_buffer('output', self(input, index, None)[-1])

    def loop_inputs(self):
        return [self.input, self.index, self.output]

    def forward(self, input, index, _):
        return input, index, self.scatter(input, index, dim=0)


class GatherOp(torch.nn.Module):

    def __init__(self,
                 num_inputs: int,
                 num_features: int,
                 num_outputs: int) -> None:
        super().__init__()
        input = torch.randn(num_inputs, num_features)
        index = torch.randint(num_inputs, (num_outputs, ))
        self.register_buffer("input", input)
        self.register_buffer("index", index)
        self.register_buffer('output', self(input, index, None)[-1])

    def loop_inputs(self):
        return [self.input, self.index, self.output]

    def forward(self, input, index, _):
        return input, index, input.index_select(dim=0, index=index)


class PerfMetrics:
    r"""Track performance metrics from:
        * recorded number of cycles
        * sizes of input / output
    Assumes IPU-BOWM2000 clock speed of 1850 MHz
    Defines an effective bandwidth from the size of the output result.
    """
    bow_clock = 1850.  # MHz

    def __init__(self, pop_model, num_repeats) -> None:
        output = pop_model.operator.output
        numels = output.numel()
        numbytes = torch.finfo(output.dtype).bits // 8
        self.out_gib = numels * numbytes / 1024**3
        self.num_repeats = num_repeats

    def update(self, cycles):
        avg_cycles = cycles / self.num_repeats
        time_us = avg_cycles / self.bow_clock
        time_s = time_us * 10**-6
        effective_bandwidth = self.out_gib / time_s

        return {
            "cycles": avg_cycles,
            "time (\u03BCs)": time_us,
            "effective bandwidth (GiB/s)": effective_bandwidth
        }


class Benchmark(torch.nn.Module):

    def __init__(self, operator: Callable, num_repeats: int) -> None:
        super().__init__()
        self.num_repeats = num_repeats
        self.operator = operator

    def forward(self):
        out = poptorch.for_loop(self.num_repeats, self.operator, 
                                self.operator.loop_inputs())[-1]
        return torch.sum(out)


def read_precomputed_benchmarks(operation):
    assert operation in ['gather', 'scatter_add']

    a = pd.read_csv(f'precomputed_results/gpu/{operation}.csv')
    b = pd.read_csv(f'precomputed_results/ipu/{operation}.csv').reset_index()

    b_time =  b['time (μs)']
    b_time = b_time.rename('IPU time (μs)')
    a_time = a['time'].rename('A100 GPU time (μs)')

    speedup = a_time / b_time

    return pd.concat([a[['num_inputs', 'num_features', 'num_outputs']], a_time, b_time, speedup], axis=1)


def legend_union(axes):
    all_handles = []
    all_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for i in range(len(labels)):
            if labels[i] not in all_labels:
                all_handles.append(handles[i])
                all_labels.append(labels[i])
    return all_handles, all_labels

def quick_benchmarks(operation):
    palette = dict([(2**p, f'C{p}') for p in range(16)])
    def draw_plot(hyperp, ax):
        ax = sns.boxplot(data=df, y=df.columns[-1], x='', ax=ax, hue=hyperp, palette=palette)
        ax.set_yticks([2**i for i in range(5)])
        ax.set_yticklabels([f'{2**i}x' for i in range(5)])
        ax.axhline(1.0, color='r') # draw horizontal line for no speedup against A100
        ax.get_legend().remove()
        ax.set_ylabel('')
        ax.set_title(hyperp)
        return ax

    df = read_precomputed_benchmarks(operation)
    df.insert(0, '', '')

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    (ax0, ax1, ax2) = axes
    plt.tight_layout()

    draw_plot('num_inputs', ax=ax0)
    draw_plot('num_features', ax=ax1)
    draw_plot('num_outputs', ax=ax2)
    
    handles, labels = legend_union(axes)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.95))
    fig.text(-0.005, 0.5, f'{operation} speedup (IPU vs A100 GPU)', ha='center', va='center', rotation='vertical')