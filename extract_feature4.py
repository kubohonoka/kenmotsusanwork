import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import wave
import struct
from torch.utils.data import DataLoader
# importするもの：パッケージ、親クラス
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms as vt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


from models import SoundFeatureExtractor


def main(feature_dim):
    learning_rate = 1e-4
    batch_size = 64
    num_epochs = 30

    device = torch.device('cuda:0')
    disable_tqdm = False

    clf = SoundFeatureExtractor(feature_dim).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)
    # データセットの読み込み
    train_dataset = pd.read_csv('./wav_keyboard_sound_train.csv').to_numpy()
    test_dataset = pd.read_csv('./wav_keyboard_sound_test.csv').to_numpy()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_dataset = wave.open(train_dataset, "rb")
    channels = train_dataset.getnchannels()
    chunk_size = train_dataset.getnframes()
    data = train_dataset.readframes(chunk_size)
    # データ前処理 # 一次配列に変換
    X = np.frombuffer(data, dtype='int16')
    # フーリエ変換により周波数データへ
    fourier = clf.fourier(X)

    for epoch_i in range(num_epochs):
        print(epoch_i)
        # 正解率の算出のための、予測と正解とをそれぞれ格納するlistを生成する。
        sum_loss = 0. # loss算出の直前だとダメ？
        # ↑ forループ内で出力される値の和を計算するとき、和を格納したい変数を0で初期化するのは、
        # ↑ プログラミングではよく用いられる手段です。for_honoka.pyに参考プログラムを記述しておきました。
        pred_all = []
        gt_all = []
        for fourier, gt in train_loader:  # gtどうしよう
            # 前のミニバッチで計算した勾配を0に戻す。
            optimizer.zero_grad()
            # clfに周波数データを渡し、ラベルを推論させる。推論結果は、lossの算出のために変数に格納しておく。
            fourier, gt = fourier.to(device), gt.to(device)
            pred = clf.predict(fourier)  # 関数とみなせるインスタンス() = __call__() → forward関数の呼び出し
            # lossを算出。
            loss = F.cross_entropy(fourier, gt)
            # lossをclfのパラメータで微分し、勾配を得る。
            loss.backward()
            # 勾配に基づき、パラメータ更新(最適化)を行う。
            optimizer.step()

    features = []
    gts = []
    clf.eval()
    with torch.no_grad():
        for features, gt in test_loader:
            features, gt = features.to(device), gt.to(device)
            feature = clf.extract(features)
            features.extend(feature.detach().cpu().numpy())
            gts.extend(gt.detach().cpu().numpy())
    features = np.concatenate([x.reshape(1, -1) for x in features], axis=0)
    tsne = TSNE(n_components=2)
    compressed = tsne.fit_transform(features)

    for gt_i in range(len(np.unique(gts))):
        where = [gt_i == x for x in gts]
        plt.scatter(compressed[where, 0], compressed[where, 1], label=str(gt_i))
    plt.legend()
    plt.savefig('./res/{}.png'.format(feature_dim))
    plt.close()


if __name__ == '__main__':
    for feature_dim in [10 * i for i in range(1, 51)]:
        main(feature_dim)
