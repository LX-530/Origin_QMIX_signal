"""
分析与对比脚本 (成果4/成果5)：读取各算法 run 的评估指标 CSV，产出
对比曲线 + 汇总表，作为 PPT「可替换结果页」的素材。

用法：
    python analyze_compare.py \
        --runs orig=./result/qmix/dynamicsignal_1 \
               fixed=./result/qmix/dynamicsignal_2 \
               cdqn=./result/cascade_dqn/dynamicsignal_1 \
        --out ./result/compare

每个 run 指向一个 save_path 目录；脚本自动在其中寻找
    historydata/cascade_eval_metrics.csv   (CDQN)
    historydata/qmix_eval_metrics.csv      (QMIX / 修正 QMIX)
也可直接给到某个 csv 文件路径。

输出：
    <out>/compare_curves.png   多指标对比曲线
    <out>/summary_table.md     关键指标汇总表（末段均值）
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 统一对比的指标（列名 -> 中文标题）
METRICS = [
    ('reward_sum', 'Episode reward'),
    ('completion', 'Completion (1=evacuated)'),
    ('evac_steps', 'Evacuation steps'),
    ('fire_cum', 'Cumulative fire exposure'),
    ('cong_cum', 'Cumulative congestion'),
]
CANDIDATE_FILES = ['cascade_eval_metrics.csv', 'qmix_eval_metrics.csv']


def find_csv(path):
    if os.path.isfile(path):
        return path
    for sub in (path, os.path.join(path, 'historydata')):
        for name in CANDIDATE_FILES:
            cand = os.path.join(sub, name)
            if os.path.isfile(cand):
                return cand
    return None


def load_csv(path):
    with open(path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    cols = {}
    if not rows:
        return cols
    for key in rows[0].keys():
        series = []
        for r in rows:
            try:
                series.append(float(r[key]))
            except (ValueError, TypeError):
                series.append(float('nan'))
        cols[key] = series
    return cols


def x_axis(cols, n):
    for key in ('env_step', 'epoch', 'episode'):
        if key in cols and len(cols[key]) == n:
            return cols[key], key
    return list(range(n)), 'eval index'


def tail_mean(series, k=10):
    vals = [v for v in series[-k:] if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='label=path 列表；path 为 run 目录或 csv 文件')
    ap.add_argument('--out', default='./result/compare')
    ap.add_argument('--tail', type=int, default=10, help='汇总表取末段多少个评估点求均值')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for item in args.runs:
        if '=' not in item:
            print('skip (need label=path):', item)
            continue
        label, path = item.split('=', 1)
        csv_path = find_csv(path)
        if csv_path is None:
            print('WARN: no metrics csv for run "{}" at {}'.format(label, path))
            continue
        cols = load_csv(csv_path)
        if cols:
            runs[label] = cols
            print('loaded {:>8}  <-  {}  ({} rows)'.format(label, csv_path, len(next(iter(cols.values())))))

    if not runs:
        print('No runs loaded. Nothing to do.')
        return

    # ---- 对比曲线 ----
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 3 * len(METRICS)))
    if len(METRICS) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, METRICS):
        plotted = False
        xlabel = 'eval index'
        for label, cols in runs.items():
            if col not in cols:
                continue
            y = cols[col]
            x, xlabel = x_axis(cols, len(y))
            ax.plot(x, y, marker='.', markersize=3, label=label)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    fig.tight_layout()
    curves_path = os.path.join(args.out, 'compare_curves.png')
    fig.savefig(curves_path, dpi=200)
    print('saved', curves_path)

    # ---- 汇总表（末段均值）----
    md = ['# 算法对比汇总（末 {} 个评估点均值）\n'.format(args.tail)]
    header = '| 算法 | ' + ' | '.join(t for _, t in METRICS) + ' |'
    sep = '|---' * (len(METRICS) + 1) + '|'
    md.append(header)
    md.append(sep)
    for label, cols in runs.items():
        cells = [label]
        for col, _ in METRICS:
            if col in cols:
                cells.append('{:.3f}'.format(tail_mean(cols[col], args.tail)))
            else:
                cells.append('-')
        md.append('| ' + ' | '.join(cells) + ' |')
    table_md = '\n'.join(md) + '\n'
    table_path = os.path.join(args.out, 'summary_table.md')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table_md)
    print('saved', table_path)
    print('\n' + table_md)


if __name__ == '__main__':
    main()
