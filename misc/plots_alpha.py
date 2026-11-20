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
    mean = []
    std = []
    for group in runs_groups:
        metric_values = []
        for run in group:
            if run.config['seed'] <= 2 and run.config.get('num_integration_steps') == num_integration_steps:
                value = run.summary.get(metric)
                if value is None:
                    # crashed / still running / never validated: no metric in the summary
                    print(f"Skipping run '{run.name}' (group {run.group}, state {run.state}): no {metric!r}")
                    continue
                metric_values.append(value)
        metric_values = np.array(metric_values, dtype=float)
        mean.append(np.mean(metric_values))
        std.append(np.std(metric_values))
    return np.asarray(mean), np.asarray(std)

def extract_runs(entity, project, lambd_, eta, alpha_max=64.0, flow_method='tuned'):
    """One list of runs per value of the swept parameter (pass a tuple for either lambd_ or eta).
    eta: 0 = conditional quantile map, 1 = total resampling. Old-name runs are included (run_groups.py)."""
    if isinstance(eta, tuple):
        configs = [(lambd_, e) for e in eta]
    elif isinstance(lambd_, tuple):
        configs = [(lam, eta) for lam in lambd_]
    else:
        configs = [(lambd_, eta)]
    return [fetch_group_runs(entity, project, flow_method, alpha_max, lam, e) for lam, e in configs]


def extract_runs_by_alpha_max(entity, project, flow_method, alpha_max_values, lambd_=0.0, eta=1.0):
    """One runs-group per alpha_max value, for a fixed flow_method/lambd_/eta, merged over the
    alpha_max spellings ('8' / '8.0') and the current and legacy group names (run_groups.py)."""
    return [fetch_group_runs(entity, project, flow_method, alpha_max, lambd_, eta)
            for alpha_max in alpha_max_values]


def grouped_barplot(alpha_labels, means, stds, methods, colors, title, ylabel, xlabel,
                     ymin=None, ymax=None, file_suffix=''):
    x = np.arange(len(alpha_labels))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots()
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, means[method], width, yerr=None, capsize=5,
               color=plt.cm.inferno(0.5) if len(methods) == 1 else colors[method],
               label=flow_method_names.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in alpha_labels])
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    if len(methods) > 1:
        ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./img/{title}_alpha_max{file_suffix}.png")
    plt.show()


def print_metric_table(alpha_max_values, means, stds, methods, decimals=1):
    print('-----')
    header = 'alpha_max & ' + ' & '.join(flow_method_names.get(m, m) for m in methods) + ' \\\\'
    print(header)
    for i, alpha_max in enumerate(alpha_max_values):
        cells = [f'$ {means[method][i]:.{decimals}f} \\pm {stds[method][i]:.{decimals}f} $' for method in methods]
        text = f'{alpha_max} & ' + ' & '.join(cells) + ' \\\\ '
        print(text)


xlabel = r'$\alpha_{max}$'
flow_methods = ('cdf_trick', 'original')
method_colors = {
    'cdf_trick': plt.cm.inferno(0.8),  # yellow
    'original': plt.cm.inferno(0.2),
}
flow_method_names = {
    'cdf_trick': 'Quantile Map', 'original': 'Vector Field'}

def alpha_max_comparison(entity, project, title, metric, alpha_max_values, ylabel, scale=1.0,
                          ymin=None, ymax=None, num_integration_steps=100, decimals=1,
                          zoom_method='cdf_trick', zoom_ymin=None, zoom_ymax=None):
    """Bar plot of both methods vs alpha_max, then a second plot with only `zoom_method`
    (saved as {title}_alpha_max_{zoom_method}.png) on a narrow y range to show its variations.
    zoom_ymin / zoom_ymax default to the min / max of the bars with a 10% margin."""
    means, stds = {}, {}
    for flow_method in flow_methods:
        runs_groups = extract_runs_by_alpha_max(entity, project, flow_method, alpha_max_values,
                                                 lambd_=0.0, eta=1.0)  # unused by these methods; = legacy groups *_0.0_0.0
        mean, std = extract_runs_metric(runs_groups, metric, num_integration_steps=num_integration_steps)
        means[flow_method] = mean * scale
        stds[flow_method] = std * scale

    print(title)
    print_metric_table(alpha_max_values, means, stds, flow_methods, decimals=decimals)

    grouped_barplot(alpha_max_values, means, stds, flow_methods, method_colors,
                     title=title, ylabel=ylabel, xlabel=xlabel, ymin=ymin, ymax=ymax)

    if zoom_method is not None:
        lo, hi = np.nanmin(means[zoom_method]), np.nanmax(means[zoom_method])
        margin = 0.1 * (hi - lo) if hi > lo else 0.05 * abs(hi)
        grouped_barplot(alpha_max_values, means, stds, (zoom_method,), method_colors,
                         title=title, ylabel=ylabel, xlabel=xlabel,
                         ymin=lo - margin if zoom_ymin is None else zoom_ymin,
                         ymax=hi + margin if zoom_ymax is None else zoom_ymax,
                         file_suffix=f'_{zoom_method}')


entity = 'yobo'
# 'original' / 'cdf_trick' ignore lambda and eta; their runs were launched with lambda=0, eta=1 (old flow_score=0).
alpha_max_comparison(entity, 'Promoter3', title='Promoter DNA', metric='val_sp-mse',
                      alpha_max_values=(8.0, 12.0, 16.0, 24.0, 32.0, 64.0), ylabel='MSE $(10^3)$', scale=1000)

alpha_max_comparison(entity, 'Melanoma', title='Melanoma', metric='val_fxd_generated_to_allseqs',
                      alpha_max_values=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 32.0, 64.0),
                      ylabel='FBD')

alpha_max_comparison(entity, 'FlyBrain', title='Fly Brain', metric='val_fxd_generated_to_allseqs',
                      alpha_max_values=(8.0, 16.0, 32.0, 64.0), ylabel='FBD')