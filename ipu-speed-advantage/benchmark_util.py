# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as cplt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
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
    speedup = speedup.rename('speedup')

    return pd.concat([a[['num_inputs', 'num_features', 'num_outputs']], a_time, b_time, speedup], axis=1)

def quick_benchmarks_3d(operation, yaw=120):
    def draw_plot(df, axes, plot_type, operation, yaw):
        cvals  = [2**i for i in range(5)]
        colors = ["#0068AA","#B5E4EB", "#FBE8AA", "#FBC3AA", "#FF6F79"]
        gc_norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(gc_norm,cvals), colors))
        gc_cmap = cplt.LinearSegmentedColormap.from_list("", tuples)
        
        z_label = ''
        
        x = np.log2(df['num_inputs'])
        y = np.log2(df['num_outputs'])
        speedup = df['speedup']
        kw = {'cmap':gc_cmap, 'norm':gc_norm, 'edgecolors': 'k', 'linewidth': .2, 'aa': 4} 
        if plot_type == 'trisurf':
            # IPU
            axes.plot_trisurf(x, y, speedup, **kw)
            # baseline
            axes.plot_trisurf(x, y, np.ones_like(speedup), **kw)
            z_label = 'Avg Speedup vs A100 ---->'
        elif plot_type == 'scatter3D':
            # IPU
            s = axes.scatter3D(x, y, speedup, c=speedup, **kw)
            # baseline
            axes.plot_trisurf(x, y, np.ones_like(speedup), **kw)
            z_label = 'Speedup vs A100 ---->'
        else:
            print('Unsupported plot type.')

        axes.set_zorder(0)
        axes.set_xlabel('num_inputs', fontsize=10, labelpad=0)
        axes.xaxis.set_major_formatter(FormatStrFormatter('$2^{%d}$'))
        axes.set_ylabel('num_outputs', fontsize=10, labelpad=0)
        axes.yaxis.set_major_formatter(FormatStrFormatter('$2^{%d}$'))
        axes.tick_params(axis='both', which='major', labelsize=7, pad=-1)
        axes.set_zlabel(z_label, fontsize=7, labelpad=-25)
        axes.set_zticks([2**i for i in range(5)])
        axes.set_zticklabels([f'{2**i}x' for i in range(5)])
        axes.set_facecolor("#FFF")
        axes.margins(0.0)

        axes.view_init(10, yaw)
        
    df = read_precomputed_benchmarks(operation)
    df.insert(0, '', '')
    
    fig = plt.figure(figsize=(9,4),dpi=300)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    draw_plot(df, axes=ax, plot_type='scatter3D', operation=operation, yaw=yaw)
    df_smooth = df.groupby(['num_inputs', 'num_outputs']).mean(numeric_only=True).reset_index() # avg feats
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    draw_plot(df_smooth, axes=ax, plot_type='trisurf', operation=operation, yaw=yaw)
    