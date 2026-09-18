

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


def extract_runs_metric(runs_groups,  metric: str, num_integration_steps=100):
    mean = []
    std = []
    for group in runs_groups:
        metric_values = []
        for run in group:
            if run.config['seed'] <= 2 and run.config.get('num_integration_steps') == num_integration_steps:
                metric_values.append(run.summary.get(metric))
        metric_values = np.array(metric_values)
        mean.append(np.mean(metric_values))
        std.append(np.std(metric_values))
    return np.asarray(mean), np.asarray(std)

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


def alpha_max_group_strings(alpha_max):
    """Group names were written with inconsistent alpha_max formatting over time
    (e.g. '8' vs '8.0'), so try every spelling that has been used."""
    if float(alpha_max).is_integer():
        return {str(int(alpha_max)), f"{float(alpha_max):.1f}"}
    return {str(alpha_max)}


def extract_runs_by_alpha_max(entity, project, flow_method, alpha_max_values, lambda_=0.0, eta=0.0):
    """One runs-group (merged across alpha_max spellings) per alpha_max value, for a fixed
    flow_method/lambda_/eta."""
    runs_groups = []
    for alpha_max in alpha_max_values:
        runs = []
        for a_str in alpha_max_group_strings(alpha_max):
            runs += list(fetch_runs(entity, project,
                                     filters={"group": f"{flow_method}_{a_str}_{lambda_}_{eta}"}))
        runs_groups.append(runs)
    return runs_groups


def grouped_barplot(alpha_labels, means, stds, methods, colors, title, ylabel, xlabel,
                     ymin=None, ymax=None):
    x = np.arange(len(alpha_labels))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots()
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, means[method], width, yerr=None, capsize=5,
               color=colors[method], label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in alpha_labels])
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./img/{title}_alpha_max.png")
    plt.show()


def print_metric_table(alpha_max_values, means, stds, methods, decimals=1):
    print('-----')
    header = 'alpha_max & ' + ' & '.join(methods) + ' \\\\'
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


def alpha_max_comparison(entity, project, title, metric, alpha_max_values, ylabel, scale=1.0,
                          ymin=None, ymax=None, num_integration_steps=100, decimals=1):
    means, stds = {}, {}
    for flow_method in flow_methods:
        runs_groups = extract_runs_by_alpha_max(entity, project, flow_method, alpha_max_values,
                                                 lambda_=0.0, eta=0.0)
        mean, std = extract_runs_metric(runs_groups, metric, num_integration_steps=num_integration_steps)
        means[flow_method] = mean * scale
        stds[flow_method] = std * scale

    print(title)
    print_metric_table(alpha_max_values, means, stds, flow_methods, decimals=decimals)

    grouped_barplot(alpha_max_values, means, stds, flow_methods, method_colors,
                     title=title, ylabel=ylabel, xlabel=xlabel, ymin=ymin, ymax=ymax)


entity = 'yobo'
# 'original' only has matching (lambda=0, eta=0) runs at these two alpha_max values in this project.
alpha_max_comparison(entity, 'Promoter3', title='Promoter DNA(bis)', metric='val_sp-mse',
                      alpha_max_values=(8.0, 12.0, 16.0, 24.0, 32.0, 64.0), ylabel='MSE $(10^3)$', scale=1000)

alpha_max_comparison(entity, 'Melanoma', title='Melanoma', metric='val_fxd_generated_to_allseqs',
                      alpha_max_values=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 32.0, 64.0),
                      ylabel='FBD')

alpha_max_comparison(entity, 'FlyBrain', title='Fly Brain', metric='val_fxd_generated_to_allseqs',
                      alpha_max_values=(8.0, 16.0, 32.0, 64.0), ylabel='FBD')