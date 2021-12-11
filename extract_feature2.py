import numpy as np
import torch
from torch.utils.data import DataLoader
# importするもの：パッケージ、親クラス
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms as vt
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


from models import FeatureExtractor


def main(feature_dim):
    learning_rate = 1e-4
    batch_size = 64
    num_epochs = 3

    device = torch.device('cuda:0')
    disable_tqdm = False

    clf = FeatureExtractor(feature_dim).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)
    train_set = MNIST(root='./', train=True, transform=vt.ToTensor(), download=True)
    eval_set = MNIST(root='./', train=False, transform=vt.ToTensor(), download=True)
    # Dateloaderとは、Datesetからサンプルを取得して、ミニバッチを作成するクラス
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, shuffle=True)

    for epoch_i in range(num_epochs):
        print(epoch_i)
        # 正解率の算出のための、予測と正解とをそれぞれ格納するlistを生成する。
        sum_loss = 0. # loss算出の直前だとダメ？
        # ↑ forループ内で出力される値の和を計算するとき、和を格納したい変数を0で初期化するのは、
        # ↑ プログラミングではよく用いられる手段です。for_honoka.pyに参考プログラムを記述しておきました。
        pred_all = []
        gt_all = []
        for img, gt in train_loader:  # (DataLoaderのインスタンスは、ここに配置されると全データをbatch_size個ずつ渡す。)
            # 前のミニバッチで計算した勾配を0に戻す。
            optimizer.zero_grad()
            # clfにimgを渡し、ラベル(数字)を推論させる。推論結果は、lossの算出のために変数に格納しておく。
            img, gt = img.to(device), gt.to(device)
            pred = clf.predict(img) # 関数とみなせるインスタンス() = __call__() → forward関数の呼び出し
            # lossを算出。
            loss = F.cross_entropy(pred, gt)
            # lossをclfのパラメータで微分し、勾配を得る。
            loss.backward()
            # 勾配に基づき、パラメータ更新(最適化)を行う。
            optimizer.step()

    features = []
    gts = []
    clf.eval()
    with torch.no_grad():
        for img, gt in eval_loader:
            img, gt = img.to(device), gt.to(device)
            feature = clf.extract(img)
            features.extend(feature.detach().cpu().numpy())
            gts.extend(gt.detach().cpu().numpy())
    features = np.concatenate([x.reshape(1,-1) for x in features], axis=0)
    tsne = TSNE(n_components=2)
    compressed = tsne.fit_transform(features)

    for gt_i in range(len(np.unique(gts))):
        where = [gt_i == x for x in gts]
        plt.scatter(compressed[where,0],compressed[where,1],label=str(gt_i))
    plt.legend()
    plt.savefig('./res/{}.png'.format(feature_dim))
    plt.close()


if __name__ == '__main__':
    for feature_dim in [10 * i for i in range(1, 51)]:
        main(feature_dim)