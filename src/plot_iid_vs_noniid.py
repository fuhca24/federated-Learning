# FedGSCS の iid と non-iid の結果を、複数シードで平均してグラフにするスクリプト
#
# 使い方（src フォルダで、run_iid_vs_noniid.py と同じオプションで実行）:
#   python plot_iid_vs_noniid.py --runs 5 --model=cnn --dataset=mnist --epochs=50
#
# 出力:
#   左のグラフ: ラウンドごとのテスト精度（線 = 平均、帯 = 平均 ± 標準偏差）
#   右のグラフ: 最終精度（論文と同じく最後の5ラウンドの平均）をシードごとに点で、平均を横線で表示

import os
import sys
import csv
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--runs', type=int, default=5)
p.add_argument('--seeds', type=str, default=None)
p.add_argument('--last', type=int, default=5, help='最終精度として平均する最後のラウンド数')
p_args, rest = p.parse_known_args()
rest = [a for a in rest if not (a.startswith('--iid') or a.startswith('--seed'))]
sys.argv = [sys.argv[0]] + rest
from options import args_parser
args = args_parser()
 
seeds = ([int(s) for s in p_args.seeds.split(',')] if p_args.seeds
         else list(range(1, p_args.runs + 1)))
 
CONDITIONS = [(1, 'IID', '#2a78d6'), (0, 'Non-IID', '#eb6834')]
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
 
 
def result_path(iid, seed):
    return ('./save/objects/FedGSCS_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_seed[{}].pkl'
            .format(args.dataset, args.model, args.epochs, args.frac, iid,
                    args.local_ep, args.local_bs, seed))
 
 
# ---- 読み込み ----
data = {}
for iid, label, _ in CONDITIONS:
    accs, used = [], []
    for s in seeds:
        path = result_path(iid, s)
        if not os.path.exists(path):
            print(f'見つかりません（飛ばします）: {path}')
            continue
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        accs.append(np.array(loaded[1], dtype=float) * 100)   # [train_loss, train_accuracy(=テスト精度)]
        used.append(s)
    if not accs:
        exit(f'{label} の結果が1つもありません。run_iid_vs_noniid.py と同じオプションか確認してください')
    n_round = min(len(a) for a in accs)
    data[label] = {'acc': np.stack([a[:n_round] for a in accs]), 'seeds': used}
 
# ---- 集計 ----
print('\n========== 結果（FedGSCS）==========')
rows = []
for label in data:
    acc = data[label]['acc']
    final = acc[:, -p_args.last:].mean(axis=1)            # シードごとの最終精度
    data[label]['final'] = final
    sd = final.std(ddof=1) if len(final) > 1 else 0.0
    print(f'{label:8s}: 最終精度（最後の{p_args.last}ラウンド平均） {final.mean():.2f}% ± {sd:.2f}  '
          f'(n={len(final)}, シード {data[label]["seeds"]})')
 
# ---- グラフ ----
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG,
                               gridspec_kw={'width_ratios': [3, 1.2]})
 
for iid, label, color in CONDITIONS:
    acc = data[label]['acc']
    x = np.arange(1, acc.shape[1] + 1)
    mean = acc.mean(axis=0)
    sd = acc.std(axis=0, ddof=1) if acc.shape[0] > 1 else np.zeros_like(mean)
    ax1.plot(x, mean, color=color, lw=2, label=f'{label} (n={acc.shape[0]})')
    ax1.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
    data[label]['mean'], data[label]['sd'] = mean, sd
 
# 線の右端に条件名と最終値を書く（近いときは上下にずらす）
ends = {label: data[label]['mean'][-1] for label in data}
hi, lo = sorted(ends, key=ends.get, reverse=True)
close = abs(ends[hi] - ends[lo]) < 6
for label, dy in [(hi, 7 if close else 0), (lo, -7 if close else 0)]:
    xe = len(data[label]['mean'])
    ax1.annotate(f'{label} {ends[label]:.1f}%', (xe, ends[label]), xytext=(6, dy),
                 textcoords='offset points', va='center', color=INK, fontsize=9)
 
ax1.set_title('FedGSCS test accuracy (mean ± 1 SD across seeds)', loc='left', color=INK, fontsize=11)
ax1.set_xlabel('Communication round')
ax1.set_ylabel('Test accuracy (%)')
ax1.set_ylim(0, 102)
ax1.set_xlim(0.5, ax1.get_xlim()[1] * 1.1)
ax1.grid(axis='y', color=GRID, lw=0.8)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(frameon=False, loc='lower right')
ax1.set_facecolor(BG)
 
# 右: シードごとの最終精度
rng = np.random.RandomState(0)
for i, (iid, label, color) in enumerate(CONDITIONS):
    final = data[label]['final']
    jitter = rng.uniform(-0.12, 0.12, size=len(final))
    ax2.scatter(np.full(len(final), i) + jitter, final, s=40, color=color,
                edgecolor=BG, linewidth=1.5, zorder=3)
    ax2.hlines(final.mean(), i - 0.28, i + 0.28, color=INK, lw=2, zorder=4)
    ax2.annotate(f'{final.mean():.1f}%', (i + 0.3, final.mean()), va='center',
                 color=INK, fontsize=9)
ax2.set_xticks([0, 1])
ax2.set_xticklabels([c[1] for c in CONDITIONS])
ax2.set_xlim(-0.5, 1.8)
ax2.set_title(f'Final accuracy\n(mean of last {p_args.last} rounds, per seed)', loc='left',
              color=INK, fontsize=11)
ax2.set_ylabel('Test accuracy (%)')
ax2.grid(axis='y', color=GRID, lw=0.8)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_facecolor(BG)
 
fig.suptitle(f'FedGSCS: IID vs Non-IID  |  {args.dataset}, {args.model}, rounds={args.epochs}, '
             f'C={args.frac}, E={args.local_ep}, B={args.local_bs}',
             color=INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout()
 
tag = '{}_{}_{}_C[{}]_E[{}]_B[{}]_runs[{}]'.format(
    args.dataset, args.model, args.epochs, args.frac, args.local_ep, args.local_bs,
    '-'.join(map(str, seeds)))
os.makedirs('./save', exist_ok=True)
png = f'./save/FedGSCS_iid_vs_noniid_{tag}.png'
fig.savefig(png, dpi=150)
 
# 平均と標準偏差を CSV にも保存（Excel などで使えるように）
csv_path = f'./save/FedGSCS_iid_vs_noniid_{tag}.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['round', 'IID_mean', 'IID_sd', 'NonIID_mean', 'NonIID_sd'])
    n_round = min(len(data['IID']['mean']), len(data['Non-IID']['mean']))
    for r in range(n_round):
        w.writerow([r + 1, f"{data['IID']['mean'][r]:.3f}", f"{data['IID']['sd'][r]:.3f}",
                    f"{data['Non-IID']['mean'][r]:.3f}", f"{data['Non-IID']['sd'][r]:.3f}"])
 
print(f'\nグラフを保存しました: {png}')
print(f'数値を保存しました:   {csv_path}')
 