#iidとnon-iidで何回も実行するスクリプト
#python iid_vs_noniid.py --runs 5 --model=cnn --dataset=mnist --rpochs=50
#--run 5 シード1~5で5回ずつ実行（合計10回）
#   --script ...      実行する main ファイル（デフォルト: federated_main_FedGSCS.py）
#   --rerun           結果ファイルがすでにあっても実行し直す（デフォルトは飛ばす）
#   それ以外のオプション（--model, --epochs, --frac など）はそのまま main に渡されます。
#   --iid と --seed はこのスクリプトが自動で付けるので指定しないでください。

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
 
seeds = ([int(s) for s in r_args.seeds.split(',')] if r_args.seeds
         else list(range(1, r_args.runs + 1)))

def result_path(iid, seed):
    return ('./save/objects/FedGSCS_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_seed[{}].pkl'
            .format(args.dataset, args.model, args.epochs, args.frac, iid,
                    args.local_ep, args.local_bs, seed))
 
 
os.makedirs('./save/logs', exist_ok=True)
jobs = [(iid, s) for iid in [1, 0] for s in seeds]
print(f'実行予定: {len(jobs)} 回（iid と non-iid × シード {seeds}）')

start_all = time.time()
for n, (iid, seed) in enumerate(jobs, 1):
    name = f'{"iid" if iid else "non-iid"} seed={seed}'
    if os.path.exists(result_path(iid, seed)) and not r_args.rerun:
        print(f'[{n}/{len(jobs)}] {name}: 結果があるので飛ばします')
        continue
    cmd = [sys.executable, r_args.script] + rest + ['--iid', str(iid), '--seed', str(seed)]
    log_path = f'./save/logs/FedGSCS_iid[{iid}]_seed[{seed}].log'
    print(f'[{n}/{len(jobs)}] {name}: 実行中...', end='', flush=True)
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
 