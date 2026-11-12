#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import torch

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


'''
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
'''
#クラスを2つずつの5グループ（0,1/2,3/4,5/6,7/8,9）に分けるnon-iid
def pair_noniid(dataset, num_users, num_groups=5):
    #端末uには「グループu%5」の画像だけ配る
    #ラベルを取り出す
    labels = np.asarray(dataset.targets)
    # 1グループあたりのクラス数（10クラスなら2）
    num_classes = int(labels.max()) + 1
    per_group = num_classes // num_groups
    #端末数が5台以下だったらエラー
    if num_users < num_groups:
        raise ValueError(f'num_users は {num_groups} 以上にしてください（今は {num_users}）')
    #グループ0～4について順に処理する
    dict_users = {}
    for g in range(num_groups):
        group_classes = list(range(g * per_group, (g + 1) * per_group))
        idxs = np.where(np.isin(labels, group_classes))[0]
        # 端末内でクラスが混ざるように
        np.random.shuffle(idxs)
        users = [u for u in range(num_users) if u % num_groups == g]
        for u, part in zip(users, np.array_split(idxs, len(users))):
            dict_users[u] = part
    return dict_users

def split_equal_size(dataset, user_classes):
    """
    user_classes = {端末番号: その端末が持つクラスのリスト} に従って画像を配る。
    ・どの端末も同じ枚数になるようにする
    ・端末の中では、持っているクラスを同じ枚数ずつにする
    ・同じ画像を2台に配ることはしない（足りない分に合わせるので、余った画像は使わない）
    """
    labels = np.asarray(dataset.targets)
    holders = {}                                   # クラス → そのクラスを持つ端末のリスト
    for u, cls in user_classes.items():
        for c in cls:
            holders.setdefault(c, []).append(u)
    # 1台あたりの枚数 T を、どのクラスも足りなくならない最大の値にする
    # （クラス c は、それを持つ端末 u に T / (u のクラス数) 枚ずつ配るので、その合計が c の枚数以下）
    T = min(np.sum(labels == c) / sum(1.0 / len(user_classes[u]) for u in us)
            for c, us in holders.items())
    T = int(T)
    parts = {u: [] for u in user_classes}
    for c, us in sorted(holders.items()):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        pos = 0
        for u in us:
            n = T // len(user_classes[u])          # 端末 u がクラス c から受け取る枚数
            parts[u].append(idx[pos:pos + n])
            pos += n
    # 端末内でクラスが混ざるように並べ替える（LocalUpdate は先頭8割を学習に使う）
    return {u: np.random.permutation(np.concatenate(parts[u])) for u in user_classes}

def noniid_pattern1(dataset, num_users):
    """
    パターン1: 1〜8番目の端末（番号 0〜7）はクラス 0〜4 だけ、
               最後の2台（10台なら 9,10番目 = 番号 8,9）はクラス 5〜9 だけを持つ。
    """
    user_classes = {u: list(range(0, 5)) if u < num_users - 2 else list(range(5, 10))
                    for u in range(num_users)}
    print('[non-iid パターン1] 番号0〜{}: クラス0〜4 / 番号{}〜{}: クラス5〜9'
          .format(num_users - 3, num_users - 2, num_users - 1))
    return split_equal_size(dataset, user_classes)
 
 
def noniid_pattern2(dataset, num_users):
    """
    パターン2: 1〜8番目の端末（番号 0〜7）はクラス 0〜4 だけ、
               最後の2台（10台なら 9,10番目 = 番号 8,9）はクラス 0〜9 すべてを持つ。
    """
    user_classes = {u: list(range(0, 5)) if u < num_users - 2 else list(range(0, 10))
                    for u in range(num_users)}
    print('[non-iid パターン2] 番号0〜{}: クラス0〜4 / 番号{}〜{}: クラス0〜9'
          .format(num_users - 3, num_users - 2, num_users - 1))
    return split_equal_size(dataset, user_classes)
 
 
def mnist_noniid(dataset, num_users):
    #non-iidを切り替えて
    #return noniid_pattern1(dataset, num_users)
    return noniid_pattern2(dataset, num_users)

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
    """
    return mnist_noniid(dataset, num_users)

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
