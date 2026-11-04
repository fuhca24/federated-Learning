#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FedGSCS を「iid」と「non-iid」の両方で、シードを変えて何回も実行するスクリプト
#
# 使い方（src フォルダで実行）:
#   python iid_vs_noniid.py --seeds 1,2,3,4,5 --model=cnn --dataset=cifar --epochs=50 \
#       --num_users=5 --frac=1.0 --local_ep=1 --r=0.1 --q_frac=0.6 --verbose=0
#
#   --runs 5          シード 1〜5 で5回ずつ実行（iid 5回 + non-iid 5回 = 合計10回）
#   --seeds 1,2,3     シードを直接指定したいときはこちら（--runs より優先）
#   --script ...      実行する main ファイル（デフォルト: federated_main_FedGSCS.py）
#   --rerun           結果ファイルがすでにあっても実行し直す（デフォルトは飛ばす）
#   それ以外のオプション（--model, --epochs, --r, --q_frac など）はそのまま main に渡されます。
#   --iid と --seed はこのスクリプトが自動で付けるので指定しないでください。
#
# 途中で止めても、もう一度同じコマンドを実行すれば、終わっていない分から再開します。

import os
import sys
import time
import argparse
import subprocess

runner = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
runner.add_argument('--runs', type=int, default=5)
runner.add_argument('--seeds', type=str, default=None)
runner.add_argument('--script', type=str, default='federated_main_FedGSCS.py')
runner.add_argument('--rerun', action='store_true')
r_args, rest = runner.parse_known_args()

for ng in ['--iid', '--seed']:
    if any(a == ng or a.startswith(ng + '=') for a in rest):
        exit(f'{ng} はこのスクリプトが自動で付けるので、指定しないでください')

# 結果ファイル名を知るために、main と同じ設定（デフォルト値込み）を読み込む
sys.argv = [sys.argv[0]] + rest
from options import args_parser
args = args_parser()
R = getattr(args, 'r', 1.0)
Q = getattr(args, 'q_frac', 0.3)

seeds = ([int(s) for s in r_args.seeds.split(',')] if r_args.seeds
         else list(range(1, r_args.runs + 1)))


def run_name(iid, seed):
    # 結果ファイルとログで同じ名前を使う（main の保存名と同じ並び）
    return ('FedGSCS_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_R[{}]_Q[{}]_seed[{}]'
            .format(args.dataset, args.model, args.epochs, args.frac, iid,
                    args.local_ep, args.local_bs, args.num_users, R, Q, seed))


def result_path(iid, seed):
    return f'./save/objects/{run_name(iid, seed)}.pkl'


os.makedirs('./save/logs', exist_ok=True)
jobs = [(iid, s) for iid in [1, 0] for s in seeds]
m = max(int(args.frac * args.num_users), 1)
q = max(int(Q * m), 1)
print(f'実行予定: {len(jobs)} 回（iid と non-iid × シード {seeds}）')
print(f'端末 {args.num_users} 台、毎ラウンドの候補 {m} 台、そのうち集約に使う {q} 台、学習データの割合 r={R}')
if m == 1:
    print('※ 候補が1台だけなので、FedGSCS の選別は働きません（--frac を大きくしてください）')
elif q == 1:
    print('※ 集約に使う端末が1台だけです（--q_frac を大きくすると複数台になります）')
print('各回の詳しい出力は ./save/logs/ に保存されます\n')

start_all = time.time()
for n, (iid, seed) in enumerate(jobs, 1):
    name = f'{"iid" if iid else "non-iid"} seed={seed}'
    if os.path.exists(result_path(iid, seed)) and not r_args.rerun:
        print(f'[{n}/{len(jobs)}] {name}: 結果があるので飛ばします')
        continue
    # -u: 出力をためずにすぐログに書く（実行中でもログで進み具合を確認できるように）
    cmd = [sys.executable, '-u', r_args.script] + rest + ['--iid', str(iid), '--seed', str(seed)]
    log_path = f'./save/logs/{run_name(iid, seed)}.log'   # オプションごとに別のログになる
    print(f'[{n}/{len(jobs)}] {name}: 実行中...（進み具合はログで確認: {log_path}）', end='', flush=True)
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if ret.returncode != 0 or not os.path.exists(result_path(iid, seed)):
        print(f' 失敗しました。ログを確認してください: {log_path}')
        sys.exit(1)
    # ログから最終のテスト精度を拾って表示
    acc = ''
    with open(log_path, encoding='utf-8', errors='ignore') as log:
        for line in log:
            if 'Test Accuracy' in line:
                acc = line.strip().split(':')[-1].strip()
    print(f' 完了（{time.time()-t0:.0f}秒、最終テスト精度 {acc}）')

print(f'\nすべて完了しました（合計 {time.time()-start_all:.0f}秒）')
print('グラフは次のコマンドで作れます（オプションは今回と同じにしてください）:')
print('  python plot_iid_vs_noniid.py ' + ' '.join(sys.argv[1:] +
      ([f'--seeds {r_args.seeds}'] if r_args.seeds else [f'--runs {r_args.runs}'])))