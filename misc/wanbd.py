"""Fetch runs from a W&B project and plot a metric against a config parameter.

Usage:
    python plot_wandb_metric.py --entity my-team --project my-project \
        --param learning_rate --metric val/accuracy

Requires: pip install wandb matplotlib
You must be logged in (`wandb login`) or have WANDB_API_KEY set.

Sampler parameters (flow_method = 'tuned'):
    eta    : 0 = conditional quantile map (deterministic), 1 = total resampling
    lambd_ : 0 = sample one class, > 0 = Dirichlet(lambda * p) weights, +inf (-1 in configs) = weighted average
Runs launched with the old argument names (sampling_score = lambda, flow_score = 1 - eta) are found too,
see run_groups.py.
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_groups import fetch_group_runs  # noqa: E402


def fetch_runs(entity: str, project: str, filters: dict):
    api = wandb.Api()
    return api.runs(f"{entity}/{project}", filters=filters)


def extract_points(runs, param: str, metric: str):
    """Return (param_values, metric_values, run_names) for runs that have both."""
    xs, ys, names = [], [], []
    for run in runs:
        # Nested keys like "optimizer.lr" are supported with dot notation
        x = run.config[param]
        y = run.summary.get(metric)
        if x is None or y is None:
            print(f"Skipping run '{run.name}': missing {param!r} or {metric!r}")
            continue
        xs.append(x)
        ys.append(y)
        names.append(run.name)
    return xs, ys, names


def extract_runs_metric(runs_groups,  metric: str, num_integration_steps=100):
    """Mean / std of `metric` over seeds 0-2 for each group of runs.

    Group names do not contain num_integration_steps, so the same group holds runs with different
    NFE (e.g. 8, 32, 100, 256 for tuned_8.0_*). Only runs with the requested NFE are used.
    """
    mean = []
    std = []
    for group in runs_groups:
        metric_values = []
        seeds = []
        for run in group:
            if run.config['seed']<=2 and run.config.get('num_integration_steps') == num_integration_steps:
                value = run.summary.get(metric)
                if value is None:
                    # crashed / still running / never validated: no metric in the summary
                    print(f"Skipping run '{run.name}' (group {run.group}, state {run.state}): no {metric!r}")
                    continue
                metric_values.append(value)
                seeds.append(run.config['seed'])
        if not metric_values:
            print(f'No run with {metric!r} and num_integration_steps={num_integration_steps} in this group')
        elif len(set(seeds)) < len(seeds):
            print(f'Warning: duplicate seeds {sorted(seeds)} (same configuration run twice) in groups '
                  f'{sorted({r.group for r in group})}')
        metric_values = np.array(metric_values, dtype=float)
        mean.append(np.mean(metric_values))
        std.append(np.std(metric_values))
    return np.asarray(mean), np.asarray(std)

def mybarplot(label, mean, std, title, ymin, ymax, xlabel=None, ylabel=None):
    bar_colors = plt.cm.inferno(np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) + 0.2)
    plt.bar(x = label, height=mean,  color=bar_colors,
            #yerr=std,capsize=5, error_kw={"capthick": 1.5, "ecolor": "black"}
            )
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"./img/{title}_{xlabel[2:-1]}.png")
    plt.show()


def extract_runs(entity, project, lambd_, eta, alpha_max=8, flow_method='tuned'):
    """One list of runs per value of the swept parameter (pass a tuple for either lambd_ or eta).

    eta: 0 = conditional quantile map, 1 = total resampling.
    lambd_: 0 = sample one class, > 0 = Dirichlet weights, math.inf (or -1) = weighted average.
    Runs logged with the old names (sampling_score / flow_score = 1 - eta) are included.
    """
    if isinstance(eta, tuple):
        configs = [(lambd_, e) for e in eta]
    elif isinstance(lambd_, tuple):
        configs = [(lam, eta) for lam in lambd_]
    else:
        configs = [(lambd_, eta)]
    return [fetch_group_runs(entity, project, flow_method, alpha_max, lam, e) for lam, e in configs]

def print_metric(mean, std, param_values, decimals=1):
    text = ''
    print('-----')
    for m, s, p in zip(mean, std, param_values):
        text = f'{p} & $ {m:.{decimals}f} \\pm {s:.{decimals}f} $ \\\\ '
        print(text)


etas = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
lambdas = (0.0, 0.1, 1.0, 10.0, 100.0, math.inf)
lambda_labels = (0.0, 0.1, 1.0, 10.0, 100.0, '+inf')

# ---------------- varying eta ----------------
entity = 'yobo'
project = 'Promoter3'
title = 'Promoter DNA(bis)'
metric = 'val_sp-mse'
ylabel = 'MSE $(10^3)$'
xlabel = '$\\eta$'
alpha_max = 32

runs_groups = extract_runs(entity, project, lambd_=0.0, eta=etas, alpha_max=alpha_max)
mean, std = extract_runs_metric(runs_groups,  metric)

print_metric(mean, std, etas, decimals=4)
mean = mean * 1000
mybarplot([str(a) for a in etas], mean, std, title=title, ymin=24, ymax=32,
          xlabel=xlabel, ylabel=ylabel)


entity = 'yobo'
project = 'FlyBrain'
title = 'Fly Brain'
metric = 'val_fxd_generated_to_allseqs'
ylabel = 'FBD'
xlabel = '$\\eta$'

runs_groups = extract_runs(entity, project, lambd_=0.0, eta=etas)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, etas)
mybarplot([str(a) for a in etas], mean, std, title=title, ymin=10, ymax=30,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'Melanoma'
title = 'Melanoma'
metric = 'val_fxd_generated_to_allseqs'
ylabel = 'FBD'
xlabel = '$\\eta$'

runs_groups = extract_runs(entity, project, lambd_=0.0, eta=etas)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, etas)
mybarplot([str(a) for a in etas], mean, std, title=title, ymin=0, ymax=50,
          xlabel=xlabel, ylabel=ylabel)


# ---------------- varying lambda, at eta = 0 (conditional quantile map) ----------------
project = 'Promoter3'
title = 'Promoter DNA'
ylabel = 'MSE $(10^3)$'
xlabel = '$\\lambda$'
metric = 'val_sp-mse'
alpha_max = 32

runs_groups = extract_runs(entity, project, lambd_=lambdas, eta=0.0, alpha_max=alpha_max)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, lambda_labels, decimals=4)
mean = mean * 1000
mybarplot([str(a) for a in lambda_labels], mean, std, title=title, ymin=24, ymax=28,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'Melanoma'
ylabel = 'FBD'
xlabel = '$\\lambda$'
metric = 'val_fxd_generated_to_allseqs'

runs_groups = extract_runs(entity, project, lambd_=lambdas, eta=0.0)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, lambda_labels)
mybarplot([str(a) for a in lambda_labels], mean, std, title=project, ymin=4, ymax=6,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'FlyBrain'
title = 'Fly Brain'
ylabel = 'FBD'
xlabel = '$\\lambda$'
metric = 'val_fxd_generated_to_allseqs'

runs_groups = extract_runs(entity, project, lambd_=lambdas, eta=0.0)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, lambda_labels)
mybarplot([str(a) for a in lambda_labels], mean, std, title=title, ymin=13, ymax=17,
          xlabel=xlabel, ylabel=ylabel)
