#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# FedGSCS（勾配の類似度による端末選択）の再現実装
#
# 使い方の例（src フォルダで実行）:
#   python federated_main_FedGSCS.py --model=cnn --dataset=cifar --epochs=50 --iid=0 \
#       --num_users=5 --frac=1.0 --local_ep=1 --r=0.1 --q_frac=0.6 --seed=1
#   --r       学習用データを何割使うか（options.py に追加。デフォルト 1.0 = 全部）
#   --q_frac  候補端末のうち、類似度の上位何割を集約に使うか（options.py に追加。デフォルト 0.3）


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


def evaluate_per_class(model, dataset, device, num_classes):
    """テストデータで、クラスごとの正解率（0〜1）を計算する"""
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(dim=1).cpu().numpy()
            y = labels.numpy()
            total += np.bincount(y, minlength=num_classes)
            correct += np.bincount(y[pred == y], minlength=num_classes)
    return (correct / np.maximum(total, 1)).tolist()


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    # コマンドライン引数の読み込み
    args = args_parser()
    exp_details(args)
    q_frac = getattr(args, 'q_frac', 0.3)
    r = getattr(args, 'r', 1.0)
    print(f'    Number of users    : {args.num_users}')
    print(f'    Data ratio (r)     : {r}')
    print(f'    q_frac             : {q_frac}\n')

    # 乱数シードの固定（--seed で指定。データの削減・分割、候補端末、ローカル学習の順番が再現される）
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # GPUの設定
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups（どのユーザーがどのデータを持つかの取得）
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # --- データ分布の確認 ---
    labels_check = np.array(train_dataset.targets)
    user_classes = {u: sorted(set(labels_check[list(user_groups[u])].tolist()))
                    for u in range(args.num_users)}
    print("\n" + "=" * 40)
    print(f"学習用データ: {len(train_dataset)} 枚 / テストデータ: {len(test_dataset)} 枚")
    for u in sorted({0, 1, args.num_users - 1}):
        u_labels = labels_check[list(user_groups[u])]
        unique_labels, counts = np.unique(u_labels, return_counts=True)
        print(f"User {u} の所持クラス: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print("=" * 40 + "\n")

    # BUILD MODEL（モデルの構築）
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        else:
            exit('Error: unrecognized dataset')

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    # copy weights（初期状態のグローバルモデルの重みを取得）
    global_weights = global_model.state_dict()

    # 勾配として使うキーを「浮動小数点のパラメータ」に限定する
    # （BatchNorm の num_batches_tracked のような整数のカウンタを除くため）
    grad_keys = [k for k in global_weights.keys()
                 if torch.is_floating_point(global_weights[k])]

    # Training
    train_loss, train_accuracy = [], []
    test_loss_hist = []   # ラウンドごとのテスト損失
    selected_hist = []    # ラウンドごとに集約に使った端末の番号
    cos_hist = []         # [追加] ラウンドごとの {端末番号: コサイン類似度}（候補になった全端末）
    print_every = 2

    # 候補端末数 m と採用数 q は毎ラウンド同じ値なので、ループの外で1回だけ計算する
    m = max(int(args.frac * args.num_users), 1)
    q = max(int(q_frac * m), 1)
    print(f'毎ラウンドの候補端末: {m} 台 / そのうち集約に使う端末: {q} 台\n')

    # 通信ラウンドのループ
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_gradients = []  # 各端末の勾配を保存するリスト
        local_sizes = []      # 各端末のデータ数（ステップ5の加重平均で使う）
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # 今回のラウンドに参加する端末をランダムに選ぶ（論文 Algorithm 1 の 5 行目）
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # 選ばれた端末ごとに、手元のデータでモデルを学習させる（ステップ2）
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            # 勾配 = 元のグローバル重み - 学習後のローカル重み を1本のベクトルにする
            grad_flat = []
            for key in grad_keys:
                diff = global_weights[key] - w[key]
                grad_flat.append(diff.view(-1).cpu())
            grad_vector = torch.cat(grad_flat)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_gradients.append(grad_vector)
            local_sizes.append(len(user_groups[idx]))

        # 平均勾配の計算（ステップ3：全部足して人数で割る＝単純平均）
        all_grads = torch.stack(local_gradients)
        mean_gradient = torch.mean(all_grads, dim=0)

        # 各端末の勾配と平均勾配のコサイン類似度（ステップ4）
        cos_scores = []
        for grad_vector in local_gradients:
            sim = torch.nn.functional.cosine_similarity(mean_gradient, grad_vector, dim=0)
            cos_scores.append(sim.item())

        # 類似度が高い上位 q 台を選ぶ
        top_q_indices = torch.topk(torch.tensor(cos_scores), k=q).indices.tolist()
        selected_real_users = [int(idxs_users[i]) for i in top_q_indices]
        selected_hist.append(selected_real_users)
        cos_hist.append({int(idxs_users[i]): float(cos_scores[i]) for i in range(len(idxs_users))})
        info = [f'{u}:{user_classes[u]}' if len(user_classes[u]) <= 3 else str(u)
                for u in selected_real_users]
        print(f"Round {epoch+1}: 採択端末(番号:持っているクラス)={info} "
              f"cos={[round(cos_scores[i], 3) for i in top_q_indices]}")

        # ステップ5：合格した端末だけを「データ量に応じた加重平均」で統合する
        selected_sizes = torch.tensor([local_sizes[i] for i in top_q_indices],
                                      dtype=torch.float32)
        weights_norm = selected_sizes / torch.sum(selected_sizes)

        new_global_weights = {}
        for key in global_weights.keys():
            # 整数のバッファ（num_batches_tracked など）は平均せず、そのままコピー
            if not torch.is_floating_point(global_weights[key]):
                new_global_weights[key] = local_weights[top_q_indices[0]][key].clone()
                continue
            new_global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
            for idx_idx, i in enumerate(top_q_indices):
                weight = weights_norm[idx_idx].item()
                new_global_weights[key] += weight * local_weights[i][key].to(device)

        global_weights = new_global_weights
        global_model.load_state_dict(global_weights)
        # ロスも選ばれた端末の平均にする
        selected_losses = [local_losses[i] for i in top_q_indices]
        loss_avg = sum(selected_losses) / len(selected_losses)
        train_loss.append(loss_avg)

        global_model.eval()  # 評価モードに切り替え
        t_acc, t_loss = test_inference(args, global_model, test_dataset)  # テストデータで評価（FedAvg側と同じ）
        train_accuracy.append(t_acc)
        # test_inference の loss はバッチ(128枚)ごとの損失の「合計」なので、バッチ数で割って平均にする
        test_loss_hist.append(t_loss / math.ceil(len(test_dataset) / 128))

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training（訓練終了後、テストデータを使った推論評価）
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    # [追加] 最終ラウンドのモデルで、クラスごとの精度を計算する
    class_acc = evaluate_per_class(global_model, test_dataset, device, args.num_classes)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- クラスごとの精度: " + ", ".join(f"{c}: {100*a:.1f}%" for c, a in enumerate(class_acc)))

    # 保存：[0 学習損失, 1 テスト精度, 2 テスト損失, 3 集約に使った端末, 4 コサイン類似度,
    #        5 最終ラウンドのクラスごとの精度, 6 各端末が持っているクラス]
    # ファイル名に端末数 N、データの割合 R、採用割合 Q、シードを入れて、条件ごとに別のファイルにする
    os.makedirs('./save/objects', exist_ok=True)
    file_name = ('./save/objects/FedGSCS_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_R[{}]_Q[{}]_seed[{}].pkl'
                 .format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                         args.local_ep, args.local_bs, args.num_users, r, q_frac, args.seed))
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_loss_hist, selected_hist,
                     cos_hist, class_acc, user_classes], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))