#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FedGSCS の iid と non-iid の結果を、複数シードで平均してグラフにするスクリプト
#
# 使い方（src フォルダで、iid_vs_noniid.py と同じオプションで実行）:
#   python plot_iid_vs_noniid.py --seeds 1,2,3,4,5 --model=cnn --dataset=cifar --epochs=50 \
#       --num_users=5 --frac=1.0 --local_ep=1 --r=0.1 --q_frac=0.6
#
# 出力（./save/ に保存）:
#   精度のグラフ: 左 = ラウンドごとのテスト精度（線 = 平均、帯 = 平均 ± 標準偏差）
#                 右 = 最終精度（最後の5ラウンドの平均）をシードごとに点で、平均を横線で表示
#   損失のグラフ（..._loss.png）: テスト損失が保存されている結果のときだけ
#   それぞれの平均・標準偏差の CSV
#   クラスごとの精度のグラフ（..._class_acc.png）: 最終ラウンドのモデルの、クラス0〜9それぞれの精度
#   ./save/tables_.../ フォルダ: 毎ラウンドのコサイン類似度の表と、各端末が選ばれた回数の表
#     （CSV と、選ばれた値に色を付けた Excel ファイル。Excel は openpyxl が入っているときだけ）
#   ※ クラスごとの精度と表は、それらを記録するようにした新しい main で実行した結果のときだけ作ります
# 日本語フォントが無い環境でも文字化けしないよう、グラフ内の表記は英語にしています。

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
R = getattr(args, 'r', 1.0)
Q = getattr(args, 'q_frac', 0.3)

seeds = ([int(s) for s in p_args.seeds.split(',')] if p_args.seeds
         else list(range(1, p_args.runs + 1)))

CONDITIONS = [(1, 'IID', '#2a78d6'), (0, 'Non-IID', '#eb6834')]
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

legacy_used = []


def result_path(iid, seed):
    # 今の形式（N, R, Q 入り）を探す。r=1.0 と q_frac=0.3（今までと同じ設定）のときだけ、
    # 以前の形式（N だけ / 端末数なし）の結果も使う
    base = './save/objects/FedGSCS_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]'.format(
        args.dataset, args.model, args.epochs, args.frac, iid, args.local_ep, args.local_bs)
    new = f'{base}_N[{args.num_users}]_R[{R}]_Q[{Q}]_seed[{seed}].pkl'
    if os.path.exists(new) or not (R == 1.0 and Q == 0.3):
        return new
    for old in [f'{base}_N[{args.num_users}]_seed[{seed}].pkl', f'{base}_seed[{seed}].pkl']:
        if os.path.exists(old):
            legacy_used.append(old)
            return old
    return new


# ---- 読み込み ----
data = {}
for iid, label, _ in CONDITIONS:
    accs, losses, used, raws = [], [], [], []
    for s in seeds:
        path = result_path(iid, s)
        if not os.path.exists(path):
            print(f'見つかりません（飛ばします）: {path}')
            continue
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        accs.append(np.array(loaded[1], dtype=float) * 100)   # [学習損失, テスト精度, テスト損失, 集約した端末]
        losses.append(np.array(loaded[2], dtype=float) if len(loaded) >= 3 else None)
        used.append(s)
        raws.append(loaded)
    if not accs:
        exit(f'{label} の結果が1つもありません。iid_vs_noniid.py と同じオプションか確認してください')
    n_round = min(len(a) for a in accs)
    data[label] = {'acc': np.stack([a[:n_round] for a in accs]), 'seeds': used, 'raw': raws,
                   'loss': (np.stack([l[:n_round] for l in losses])
                            if all(l is not None for l in losses) else None)}

if legacy_used:
    print(f'※ 古い形式のファイル名の結果を {len(legacy_used)} 個使っています。')
    print(f'  --num_users={args.num_users} で実行した結果かどうか、自分で確認してください。')

# ---- 集計 ----
print('\n========== 結果（FedGSCS）==========')
for label in data:
    acc = data[label]['acc']
    final = acc[:, -p_args.last:].mean(axis=1)            # シードごとの最終精度
    data[label]['final'] = final
    sd = final.std(ddof=1) if len(final) > 1 else 0.0
    print(f'{label:8s}: 最終精度（最後の{p_args.last}ラウンド平均） {final.mean():.2f}% ± {sd:.2f}  '
          f'(n={len(final)}, シード {data[label]["seeds"]})')

title_conf = (f'{args.dataset}, {args.model}, rounds={args.epochs}, N={args.num_users}, '
              f'C={args.frac}, E={args.local_ep}, B={args.local_bs}, r={R}, q={Q}')


def hide_top_right(ax):
    for side in ('top', 'right'):   # 古い matplotlib でも動く書き方
        ax.spines[side].set_visible(False)


def label_ends(ax, ends, fmt, gap):
    # 線の右端に条件名と最終値を書く（近いときは上下にずらす）
    hi, lo = sorted(ends, key=ends.get, reverse=True)
    close = abs(ends[hi] - ends[lo]) < gap
    for label, dy in [(hi, 7 if close else 0), (lo, -7 if close else 0)]:
        xe = len(data[label]['acc'][0])
        ax.annotate(f'{label} {fmt(ends[label])}', (xe, ends[label]), xytext=(6, dy),
                    textcoords='offset points', va='center', color=INK, fontsize=9)


# ---- 精度のグラフ ----
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
label_ends(ax1, {l: data[l]['mean'][-1] for l in data}, lambda v: f'{v:.1f}%', 6)
ax1.set_title('FedGSCS test accuracy (mean ± 1 SD across seeds)', loc='left', color=INK, fontsize=11)
ax1.set_xlabel('Communication round')
ax1.set_ylabel('Test accuracy (%)')
ax1.set_ylim(0, 102)
ax1.set_xlim(0.5, ax1.get_xlim()[1] * 1.1)
ax1.grid(axis='y', color=GRID, lw=0.8)
hide_top_right(ax1)
ax1.legend(frameon=False, loc='lower right')
ax1.set_facecolor(BG)

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
hide_top_right(ax2)
ax2.set_facecolor(BG)
fig.suptitle(f'FedGSCS: IID vs Non-IID  |  {title_conf}', color=INK, fontsize=10, x=0.01, ha='left')
fig.tight_layout()

tag = '{}_{}_{}_C[{}]_E[{}]_B[{}]_N[{}]_R[{}]_Q[{}]_runs[{}]'.format(
    args.dataset, args.model, args.epochs, args.frac, args.local_ep, args.local_bs,
    args.num_users, R, Q, '-'.join(map(str, seeds)))
os.makedirs('./save', exist_ok=True)
png = f'./save/FedGSCS_iid_vs_noniid_{tag}.png'
fig.savefig(png, dpi=150)


def save_csv(path, key_mean, key_sd, digits):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['round', 'IID_mean', 'IID_sd', 'NonIID_mean', 'NonIID_sd'])
        n_round = min(len(data['IID'][key_mean]), len(data['Non-IID'][key_mean]))
        for r_ in range(n_round):
            w.writerow([r_ + 1] + [f'{data[l][k][r_]:.{digits}f}'
                                   for l in ['IID', 'Non-IID'] for k in (key_mean, key_sd)])


csv_path = f'./save/FedGSCS_iid_vs_noniid_{tag}.csv'
save_csv(csv_path, 'mean', 'sd', 3)
print(f'\nグラフを保存しました: {png}')
print(f'数値を保存しました:   {csv_path}')

# ---- 損失のグラフ（テスト損失が保存されている場合だけ）----
if all(data[label]['loss'] is not None for label in data):
    fig2, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    print('\n---------- テスト損失 ----------')
    for iid, label, color in CONDITIONS:
        loss = data[label]['loss']
        x = np.arange(1, loss.shape[1] + 1)
        mean = loss.mean(axis=0)
        sd = loss.std(axis=0, ddof=1) if loss.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(x, mean, color=color, lw=2, label=f'{label} (n={loss.shape[0]})')
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
        data[label]['loss_mean'], data[label]['loss_sd'] = mean, sd
        final = loss[:, -p_args.last:].mean(axis=1)
        fsd = final.std(ddof=1) if len(final) > 1 else 0.0
        print(f'{label:8s}: 最終テスト損失（最後の{p_args.last}ラウンド平均） {final.mean():.4f} ± {fsd:.4f}')
    ymax = max(np.max(data[l]['loss_mean'] + data[l]['loss_sd']) for l in data)
    label_ends(ax, {l: data[l]['loss_mean'][-1] for l in data}, lambda v: f'{v:.3f}', 0.06 * ymax)
    ax.set_title('FedGSCS test loss (mean ± 1 SD across seeds, lower is better)', loc='left',
                 color=INK, fontsize=11)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Test loss (NLL, per batch)')
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, ax.get_xlim()[1] * 1.12)
    ax.grid(axis='y', color=GRID, lw=0.8)
    hide_top_right(ax)
    ax.legend(frameon=False, loc='upper right')
    ax.set_facecolor(BG)
    # タイトルが重ならないように、条件はグラフの下に書く
    fig2.text(0.01, 0.01, title_conf, color=MUTED, fontsize=8, ha='left', va='bottom')
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    png_loss = f'./save/FedGSCS_iid_vs_noniid_{tag}_loss.png'
    fig2.savefig(png_loss, dpi=150)
    csv_loss = f'./save/FedGSCS_iid_vs_noniid_{tag}_loss.csv'
    save_csv(csv_loss, 'loss_mean', 'loss_sd', 5)
    print(f'\n損失のグラフを保存しました: {png_loss}')
    print(f'損失の数値を保存しました:   {csv_loss}')
else:
    print('\n※ 損失のグラフは作りませんでした。テスト損失が入っていない（古い版で実行した）結果が含まれています。')


# ======================================================================
# 追加：クラスごとの精度（最終ラウンド）と、コサイン類似度・選ばれた回数の表
# ======================================================================
CIFAR_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def fmt_range(nums):
    """[0,1,2,3,4] → '0-4'、[0,2,5] → '0,2,5' のように短く書く"""
    nums = sorted(nums)
    if len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1)):
        return f'{nums[0]}-{nums[-1]}'
    return ','.join(map(str, nums))


def describe_split(user_classes):
    """'clients 0-7: classes 0-4 | clients 8-9: classes 5-9' のような説明を作る"""
    groups = {}
    for u in sorted(user_classes):
        groups.setdefault(tuple(user_classes[u]), []).append(u)
    return ' | '.join(f'clients {fmt_range(us)}: classes {fmt_range(list(c))}'
                      for c, us in sorted(groups.items(), key=lambda kv: kv[1][0]))


has_class_acc = all(len(raw) >= 7 for l in data for raw in data[l]['raw'])
has_cos = all(len(raw) >= 5 for l in data for raw in data[l]['raw'])
if not (has_class_acc and has_cos):
    print('\n※ クラスごとの精度と、コサイン類似度の表は作りませんでした。'
          '\n  それらを記録するようにした新しい federated_main_FedGSCS.py で実行し直した結果が必要です。')

# ---- クラスごとの精度のグラフ（最終ラウンド）----
if has_class_acc:
    nc = args.num_classes
    fig3, ax = plt.subplots(figsize=(11, 5.4), facecolor=BG)
    width = 0.38
    rows = []
    print('\n---------- 最終ラウンドのクラスごとの精度 ----------')
    for k, (iid, label, color) in enumerate(CONDITIONS):
        ca = np.array([raw[5] for raw in data[label]['raw']], dtype=float) * 100   # シード × クラス
        mean = ca.mean(axis=0)
        sd = ca.std(axis=0, ddof=1) if ca.shape[0] > 1 else np.zeros(nc)
        x = np.arange(nc) + (k - 0.5) * width
        ax.bar(x, mean, width * 0.92, color=color, label=f'{label} (n={ca.shape[0]})', zorder=3)
        if ca.shape[0] > 1:
            ax.errorbar(x, mean, yerr=sd, fmt='none', ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        for xi, v, e in zip(x, mean, sd):
            ax.annotate(f'{v:.0f}', (xi, v + e), xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, color=INK)
        rows.append((label, mean, sd))
        print(f'{label:8s}: ' + '  '.join(f'{c}:{m:5.1f}%' for c, m in enumerate(mean)))
    ticks = [f'{c}\n{CIFAR_NAMES[c]}' if args.dataset == 'cifar' else str(c) for c in range(nc)]
    ax.set_xticks(np.arange(nc))
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_xlabel('Class')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    hide_top_right(ax)
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.set_facecolor(BG)
    n_round = len(data['IID']['acc'][0])
    ax.set_title(f'FedGSCS per-class test accuracy at the final round (round {n_round}), '
                 f'mean{" ± 1 SD" if len(seeds) > 1 else ""} across seeds', loc='left', color=INK, fontsize=11)
    split_text = 'Non-IID split: ' + describe_split(data['Non-IID']['raw'][0][6])
    fig3.text(0.01, 0.045, split_text, color=INK, fontsize=8.5, ha='left', va='bottom')
    fig3.text(0.01, 0.01, title_conf, color=MUTED, fontsize=8, ha='left', va='bottom')
    fig3.tight_layout(rect=(0, 0.08, 1, 1))
    png_cls = f'./save/FedGSCS_iid_vs_noniid_{tag}_class_acc.png'
    fig3.savefig(png_cls, dpi=150)
    csv_cls = f'./save/FedGSCS_iid_vs_noniid_{tag}_class_acc.csv'
    with open(csv_cls, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'name', 'IID_mean', 'IID_sd', 'NonIID_mean', 'NonIID_sd'])
        for c in range(nc):
            name = CIFAR_NAMES[c] if args.dataset == 'cifar' else str(c)
            w.writerow([c, name] + [f'{v[c]:.2f}' for _, m, sd in rows for v in (m, sd)])
    print(f'\nクラスごとの精度のグラフを保存しました: {png_cls}')
    print(f'クラスごとの精度の数値を保存しました:   {csv_cls}')

# ---- コサイン類似度と選ばれた回数の表 ----
if has_cos:
    table_dir = f'./save/tables_{tag}'
    os.makedirs(table_dir, exist_ok=True)
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
    except ImportError:
        openpyxl = None
        print('\n※ openpyxl が無いので、色付きの Excel ファイルは作りません（CSV は作ります）。'
              '\n  欲しい場合は pip install openpyxl を実行してください。')

    for iid, label, color in CONDITIONS:
        fname = label.replace('-', '')                 # IID / NonIID
        user_classes = data[label]['raw'][0][6]
        clients = sorted(user_classes)
        col_names = [f'client{u} [{fmt_range(user_classes[u])}]' for u in clients]
        wb = openpyxl.Workbook() if openpyxl else None
        if wb:
            wb.remove(wb.active)
        count = {u: [] for u in clients}               # シードごとの選ばれた回数
        cos_all = {u: [] for u in clients}             # 全ラウンドのコサイン類似度
        cos_sel = {u: [] for u in clients}             # 選ばれたときのコサイン類似度

        for seed, raw in zip(data[label]['seeds'], data[label]['raw']):
            selected_hist, cos_hist = raw[3], raw[4]
            # 1) 毎ラウンドのコサイン類似度（CSV）
            path = os.path.join(table_dir, f'cos_{fname}_seed{seed}.csv')
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['round'] + col_names + ['selected'])
                for rd, (cos_r, sel) in enumerate(zip(cos_hist, selected_hist), 1):
                    w.writerow([rd] + [f'{cos_r[u]:.4f}' if u in cos_r else '' for u in clients]
                               + [' '.join(map(str, sel))])
            for u in clients:
                count[u].append(sum(u in sel for sel in selected_hist))
                for cos_r, sel in zip(cos_hist, selected_hist):
                    if u in cos_r:
                        cos_all[u].append(cos_r[u])
                        if u in sel:
                            cos_sel[u].append(cos_r[u])
            # 2) 同じ表を Excel に（選ばれた値に色と太字を付ける）
            if wb:
                ws = wb.create_sheet(f'seed{seed}')
                ws.append(['round'] + col_names + ['selected'])
                hl = PatternFill('solid', fgColor='FBD5C4')
                for rd, (cos_r, sel) in enumerate(zip(cos_hist, selected_hist), 1):
                    ws.append([rd] + [round(cos_r[u], 4) if u in cos_r else None for u in clients]
                              + [' '.join(map(str, sel))])
                    for j, u in enumerate(clients):
                        if u in sel:
                            cell = ws.cell(row=rd + 1, column=j + 2)
                            cell.fill = hl
                            cell.font = Font(bold=True)
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(wrap_text=True, horizontal='center')
                ws.freeze_panes = 'B2'
                for j in range(len(clients) + 2):
                    ws.column_dimensions[openpyxl.utils.get_column_letter(j + 1)].width = 13

        # 3) 端末ごとの選ばれた回数（全シードまとめ）
        n_rounds = len(data[label]['raw'][0][3])
        header = (['client', 'classes'] + [f'seed{s}' for s in data[label]['seeds']]
                  + ['mean_count', f'select_rate(/{n_rounds})', 'mean_cos', 'mean_cos_when_selected'])
        summary = []
        for u in clients:
            mc = float(np.mean(count[u]))
            summary.append([u, fmt_range(user_classes[u])] + count[u]
                           + [round(mc, 2), f'{100 * mc / n_rounds:.1f}%',
                              round(float(np.mean(cos_all[u])), 4) if cos_all[u] else '',
                              round(float(np.mean(cos_sel[u])), 4) if cos_sel[u] else ''])
        path = os.path.join(table_dir, f'selection_count_{fname}.csv')
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(summary)
        if wb:
            ws = wb.create_sheet('selection_count', 0)
            ws.append(header)
            for row in summary:
                ws.append(row)
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for j in range(len(header)):
                ws.column_dimensions[openpyxl.utils.get_column_letter(j + 1)].width = 14
            wb.save(os.path.join(table_dir, f'cos_{fname}.xlsx'))

        print(f'\n---------- {label}: 各端末が選ばれた回数（{n_rounds}ラウンド中、シード平均）----------')
        for row in summary:
            print(f'  client{row[0]} [{row[1]}]: {row[-4]:5.1f}回 ({row[-3]})  '
                  f'平均cos {row[-2]}  選ばれたときの平均cos {row[-1]}')
    print(f'\nコサイン類似度と選ばれた回数の表を保存しました: {table_dir}/')