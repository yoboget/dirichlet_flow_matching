"""Fetch runs from a W&B project and plot a metric against a config parameter.

Usage:
    python plot_wandb_metric.py --entity my-team --project my-project \
        --param learning_rate --metric val/accuracy

Requires: pip install wandb matplotlib
You must be logged in (`wandb login`) or have WANDB_API_KEY set.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import wandb


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


def extract_runs_metric(runs_groups,  metric: str):
    mean = []
    std = []
    for group in runs_groups:
        metric_values = []
        for run in group:
            if run.config['seed']<=2:
                metric_values.append(run.summary.get(metric))
        metric_values = np.array(metric_values)
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
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"./img/{title}_{xlabel[2:-1]}.png")
    plt.show()


def extract_runs(entity, project, lambda_, eta, alpha_max=64.0,
                 nfe=100, flow_method='tuned'):
    runs_groups = []
    if isinstance(eta,  tuple):
        for i, eta in enumerate(eta):
            runs_groups.append(fetch_runs(entity, project, filters={"group": f"{flow_method}_{alpha_max}_{lambda_}_{eta}"
                                                                    }))
    elif isinstance(lambda_, tuple):
        for i, lambda_ in enumerate(lambda_):
            runs_groups.append(fetch_runs(entity, project, filters={"group": f"{flow_method}_{alpha_max}_{lambda_}_{eta}"
                                                                    }))
    return runs_groups

def print_metric(mean, std, param_values, decimals=1):
    text = ''
    print('-----')
    for m, s, p in zip(mean, std, param_values):
        text = f'{p} & $ {m:.{decimals}f} \pm {s:.{decimals}f} $ \\\ '
        print(text)

entity = 'yobo'
project = 'Promoter3'
title = 'Promoter DNA(bis)'
metric = 'val_sp-mse'
param = 'sampling_score'
ylabel = 'MSE $(10^3)$'
xlabel = '$\eta$'
param_values = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)[::-1]
labels = (1.0, 0.95, 0.75, 0.50, 0.25, 0.05, 0.0)[::-1]

runs_groups = extract_runs(entity, project, lambda_=0.0, eta = param_values)
mean, std = extract_runs_metric(runs_groups,  metric)

print_metric(mean, std, labels, decimals=4)
mean = mean * 1000
mybarplot([str(a) for a in labels], mean, std, title=title, ymin=24, ymax=43,
          xlabel=xlabel, ylabel=ylabel)


entity = 'yobo'
project = 'FlyBrain'
title = 'Fly Brain'
metric = 'val_fxd_generated_to_allseqs'
param = 'sampling_score'
ylabel = 'FBD'
xlabel = '$\eta$'
param_values = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)[::-1]
labels = (1.0, 0.95, 0.75, 0.5, 0.25, 0.05, 0.0)[::-1]

runs_groups = extract_runs(entity, project, lambda_=0.0, eta = param_values)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, labels)
mybarplot([str(a) for a in labels], mean, std, title=title, ymin=10, ymax=30,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'FlyBrain'
title = 'Fly Brain ($\lambda=+$inf)'
metric = 'val_fxd_generated_to_allseqs'
param = 'sampling_score'
ylabel = 'FBD'
xlabel = '$\eta$'
param_values = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)[::-1]
labels = (1.0, 0.95, 0.75, 0.5, 0.25, 0.05, 0.0)[::-1]

runs_groups = extract_runs(entity, project, lambda_=-1.0, eta = param_values)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, labels)
mybarplot([str(a) for a in labels], mean, std, title=title, ymin=10, ymax=30,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'Melanoma'
title = 'Melanoma'
metric = 'val_fxd_generated_to_allseqs'
param = 'sampling_score'
ylabel = 'FBD'
xlabel = '$\eta$'
param_values = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)[::-1]
labels = (1.0, 0.95, 0.75, 0.5, 0.25, 0.05, 0.0)[::-1]

runs_groups = extract_runs(entity, project, lambda_=0.0, eta = param_values)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, labels)
mybarplot([str(a) for a in labels], mean, std, title=title, ymin=0, ymax=50,
          xlabel=xlabel, ylabel=ylabel)


project = 'Promoter3'
title = 'Promoter DNA'
ylabel = 'MSE $(10^3)$'
xlabel = '$\lambda$'
metric = 'val_sp-mse'
param = 'sampling_score'
alpha_max = 64.0
param_values = (0.0, 0.1, 1.0, 10.0, 100.0, -1.0)
labels = (0.0, 0.1, 1.0, 10.0, 100.0, '+inf')

runs_groups = extract_runs(entity, project, lambda_=param_values, eta = 1.0,
                           alpha_max=alpha_max)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, labels, decimals=4)
mean = mean * 1000
mybarplot([str(a) for a in labels], mean, std, title=title, ymin=26, ymax=29,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'Melanoma'
ylabel = 'FBD'
xlabel = '$\lambda$'
metric = 'val_fxd_generated_to_allseqs'

param = 'flow_score'
param_values = (0.0, 0.1, 1.0, 10.0, 100.0, -1.0)
labels = (0.0, 0.1, 1.0, 10.0, 100.0, '+inf')

runs_groups = extract_runs(entity, project, lambda_=param_values, eta = 0.0)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, labels)
mybarplot([str(a) for a in labels], mean, std, title=project, ymin=4, ymax=6,
          xlabel=xlabel, ylabel=ylabel)

entity = 'yobo'
project = 'FlyBrain'
title = 'Fly Brain'
ylabel = 'FBD'
xlabel = '$\lambda$'
metric = 'val_fxd_generated_to_allseqs'
param = 'flow_score'
param_values = (0.0, 0.1, 1.0, 10.0, 100.0, -1.0)
labels = (0.0, 0.1, 1.0, 10.0, 100.0, '+inf')

runs_groups = extract_runs(entity, project, lambda_=param_values, eta = 1.0)
mean, std = extract_runs_metric(runs_groups,  metric)
print_metric(mean, std, labels)
mybarplot([str(a) for a in labels], mean, std, title=title, ymin=13, ymax=17,
          xlabel=xlabel, ylabel=ylabel)
