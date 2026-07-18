#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
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


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    # コマンドライン引数の読み込み
    args = args_parser()
    exp_details(args)
    # GPUの設定
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups（どのユーザーがどのデータを持つかの取得）
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL（モデルの構築）
    # 指定された設定（args.modelとargs.dataset）に応じて、適切なニューラルネットワークを生成する

    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

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
    # ここで作られるglobal_modelが、サーバー側で管理される「全体の中心となる共有モデル」になる
    # copy weights（初期状態のグローバルモデルの重みを取得）
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    #通信ラウンドのループ
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_gradients = []#各端末の勾配を保存するリスト
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        #今回のラウンドに参加するユーザーをランダムに選別
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        #選ばれたユーザーごとに、データを渡して手元でモデルを訓練させる。
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            #グローバルモデルのコピーを各ユーザーに渡し、ローカルで訓練
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            #勾配の計算
            #state_dictの各レイヤーのパラメーター差分を計算し、1次元にフラット化して結合する。
            grad_flat=[]
            for key in global_weights.keys():
                #勾配＝元のグローバル重み-更新後のローカル重み
                diff=global_weights[key]-w[key]
                grad_flat.append(diff.view(-1).cpu())#一次元に引き伸ばしてcpuへ
            #全レイヤーの勾配を1本の長いベクトルにする
            grad_vector=torch.cat(grad_flat)
            #学習後の重みと損失をリストに保存
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_gradients.append(grad_vector)#勾配ベクトルをリストに保存

        #平均勾配の計算
        all_grads=torch.stack(local_gradients)
        mean_gradient=torch.mean(all_grads,dim=0)

        #各端末のコサイン類似度を計算
        cos_scores=[]
        for grad_vector in local_gradients:
            #コサイン類似度の計算
            sim = torch.nn.functional.cosine_similarity(mean_gradient, grad_vector, dim=0)
            cos_scores.append(sim.item())
        
        #類似度が高い上位q個の端末の厳選
        #何個にしたらいいだろう とりあえず上位50%にするよ
        q=max(int(0.5*m),1)
        top_q_indices = torch.topk(torch.tensor(cos_scores), k=q).indices.tolist()

        #合格した端末のデータだけを抽出
        selected_weights = [local_weights[i] for i in top_q_indices]
        selected_losses = [local_losses[i] for i in top_q_indices]

        #厳選された優秀の重みだけで合体
        global_weights = average_weights(selected_weights)
        global_model.load_state_dict(global_weights)

        # ロス計算も選ばれた端末の平均にする
        loss_avg = sum(selected_losses) / len(selected_losses)
        train_loss.append(loss_avg)
        """
        # update global weights（各ユーザーから集まった重みの平均を計算）
        global_weights = average_weights(local_weights)

        # update global weights（平均化した重みを、大元のグローバルモデルに反映）
        global_model.load_state_dict(global_weights)
        # 損失の平均を計算して記録
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        """
        # Calculate avg training accuracy over all users at every epoch
        # すべてのユーザーにおける平均トレーニング精度を計算
        list_acc, list_loss = [], []
        global_model.eval()#評価モードに切り替え
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds（指定されたラウンドごとに進捗をプリント）
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training（訓練終了後、テストデータを使った推論評価）
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # 学習結果（損失と精度の推移）をローカルに保存
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
