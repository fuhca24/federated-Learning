import os
import pickle
import matplotlib.pyplot as plt
from options import args_parser

if __name__ == '__main__':
    args = args_parser()

    # ファイルパスの指定
    name_pattern = '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs
    )
    path_fedavg = os.path.join('./save/objects', f'FedAvg_{name_pattern}')
    path_fedgscs = os.path.join('./save/objects', f'FedGSCS_{name_pattern}')

    # データ読み込み
    with open(path_fedavg, 'rb') as f:
        fedavg_loss, fedavg_acc = pickle.load(f)

    with open(path_fedgscs, 'rb') as f:
        fedgscs_loss, fedgscs_acc = pickle.load(f)

    epochs = range(1, len(fedavg_acc) + 1)

    # グラフ描画（1行2列）
    plt.figure(figsize=(12, 5))

    # ① 精度 (Accuracy) 比較
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [a * 100 for a in fedavg_acc], 'o-', label='FedAvg', color='blue')
    plt.plot(epochs, [a * 100 for a in fedgscs_acc], 's--', label='FedGSCS', color='red')
    plt.title(f'Accuracy vs Communication Rounds ({args.dataset.upper()})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Train Accuracy (%)')
    plt.grid(True, linestyle=':')
    plt.legend()

    # ② 損失 (Loss) 比較
    plt.subplot(1, 2, 2)
    plt.plot(epochs, fedavg_loss, 'o-', label='FedAvg', color='blue')
    plt.plot(epochs, fedgscs_loss, 's--', label='FedGSCS', color='red')
    plt.title(f'Loss vs Communication Rounds ({args.dataset.upper()})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Training Loss')
    plt.grid(True, linestyle=':')
    plt.legend()

    # 保存
    os.makedirs('./save', exist_ok=True)
    save_path = f'./save/compare_{args.dataset}_{args.model}_E[{args.epochs}].png'
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Saved comparison plot to: {save_path}')
    plt.show()