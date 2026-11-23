"""Sampling time of 'original' (Vector Field) vs 'cdf_trick' (Quantile Map) from W&B, per hardware.

Logged by GeneralModule.time_sampling (lightning_modules/general_module.py):
    val_sampling_time   : wall time to sample one validation batch, averaged over the batches
                          (weighted by batch size)
    total_sampling_time : summed over all validation batches. Only in runs launched after it was added
                          (None otherwise).
Runs from before the CUDA synchronisation / the BetaTable-on-GPU fix are timed with the
Quantile Map transport step on the CPU, so their time depends on the number of CPU cores (cpus).

Usage:  python sampling_time.py            (or import sampling_time_table)
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_groups import fetch_group_runs  # noqa: E402

flow_method_names = {'original': 'Vector Field', 'cdf_trick': 'Quantile Map'}


def fetch_sampling_times(entity, project, flow_method, alpha_max=8, num_integration_steps=100,
                         seeds=(0, 1, 2), lambd_=0.0, eta=1.0):
    """One dict per finished run: seed, hardware (host, gpu, cpus) and the logged sampling times.
    'original' / 'cdf_trick' ignore lambda/eta; their runs are in the legacy groups *_0.0_0.0."""
    out = []
    for run in fetch_group_runs(entity, project, flow_method, alpha_max, lambd_, eta):
        c, md = run.config, run.metadata or {}
        if c.get('num_integration_steps') != num_integration_steps or c.get('seed') not in seeds:
            continue
        per_batch = run.summary.get('val_sampling_time')
        total = run.summary.get('total_sampling_time')
        if per_batch is None and total is None:
            print(f"Skipping run '{run.name}' ({run.state}): no sampling time")
            continue
        out.append(dict(
            run=run.name, seed=c['seed'], created=run.created_at, batch_size=c.get('batch_size'),
            host=md.get('host'), gpu=md.get('gpu'), cpus=int((md.get('slurm') or {}).get('cpus_per_task', 0)) or None,
            per_batch=None if per_batch is None else float(per_batch),
            total=None if total is None else float(total)))
    return out


def total_from_val_csv(path):
    """Exact total sampling time of a run from its workdir/<run>/val_<step>.csv, for runs that did not log
    total_sampling_time to W&B (the on_validation_epoch_end overrides did not log it before 2026-11-23).

    The CSV has one row per sequence; the column val_sampling_time holds the time of the batch the sequence
    was in, repeated for each sequence of the batch, so the total is the sum over the batches of their time.
    """
    import pandas as pd
    t = pd.read_csv(path)['val_sampling_time']
    batch_start = t.ne(t.shift())          # consecutive rows with the same value belong to one batch
    return float(t[batch_start].sum()), int(batch_start.sum())   # (total seconds, number of batches)


def _hardware(r):
    return f"{r['gpu']} ({r['host']}, {r['cpus']} CPU)"


def sampling_time_table(entity, experiments, methods=('original', 'cdf_trick'), alpha_max=8,
                        num_integration_steps=100, gpu=None, cpus=None, created_after=None, quantity='total',
                        decimals=2):
    """Mean +- std over seeds of the sampling time, one row per (experiment, hardware).

    quantity: 'total' = total_sampling_time (all validation batches, i.e. the whole test set),
              'per_batch' = val_sampling_time (one batch). Runs without that quantity are left out.

    experiments: list of (project, display title).
    gpu / cpus : keep only runs whose GPU name contains `gpu` / with `cpus` CPU cores (None = all, and the
                 rows are split per hardware so that times from different machines are never averaged).
    created_after: ISO date, e.g. '2026-11-20': keep only runs created on or after it (e.g. only the runs
                   with BetaTable on the GPU and the synchronised timer, see run_time_*.sh).
    If a seed was run twice on the same hardware, the most recent run is kept.
    Returns {(title, hardware): {method: (mean, std, n)}}.
    """
    assert quantity in ('total', 'per_batch'), quantity
    table = defaultdict(dict)
    for project, title in experiments:
        for method in methods:
            by_hw = defaultdict(dict)
            for r in sorted(fetch_sampling_times(entity, project, method, alpha_max, num_integration_steps),
                            key=lambda r: r['created']):
                if (gpu is not None and gpu not in (r['gpu'] or '')) or (cpus is not None and r['cpus'] != cpus) \
                        or (created_after is not None and r['created'] < created_after) or r[quantity] is None:
                    continue
                by_hw[_hardware(r)][r['seed']] = r          # later run overwrites the same seed
            for hw, runs in by_hw.items():
                v = np.array([r[quantity] for r in runs.values()])
                table[(title, hw)][method] = (v.mean(), v.std(), len(v))

    what = 'total sampling time (whole test set)' if quantity == 'total' else 'sampling time per batch'
    print(f'{what} (s), alpha_max = {alpha_max}, {num_integration_steps} integration steps')
    print('Experiment & Hardware & ' + ' & '.join(flow_method_names.get(m, m) for m in methods)
          + ' \\\\  (mean +- std over n seeds)')
    for (title, hw), res in table.items():
        cells = []
        for m in methods:
            if m not in res:
                cells.append('--')
                continue
            mean, std, n = res[m]
            cells.append(f'$ {mean:.{decimals}f} \\pm {std:.{decimals}f} $ (n={n})')
        print(f'{title} & {hw} & ' + ' & '.join(cells) + ' \\\\')
    return dict(table)


if __name__ == '__main__':
    experiments = [('Promoter3', 'Promoter'), ('Melanoma', 'Melanoma'), ('FlyBrain', 'Fly Brain')]
    # runs from run_time_*.sh (BetaTable on GPU, synchronised timer, total_sampling_time logged)
    sampling_time_table('yobo', experiments, gpu='RTX 3090', cpus=1, created_after='2026-11-20')
    # old runs (before the fix) only have the per-batch time:
    # sampling_time_table('yobo', experiments, quantity='per_batch')
