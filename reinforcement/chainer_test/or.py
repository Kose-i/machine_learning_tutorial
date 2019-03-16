#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functinos as F
import chainer.link as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2,3)
            self.l2 = L.Linear(3,2)
    def __call__(self, X):
        h1 = F.relu(self.l1(X))
        y = self.l2(h1)
        return y

if __name__=='__main__':
    epoch = 100
    batchsize = 4

# データの作成
    trainX = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=float32)
    trainy = np.array([0,1,1,1], dtype=np.int32)
    train = chainer.datasets.TupleDataset(trainX, trainy)
    test = chainer.datasets.TupleDataset(trainX, trainy)

# ニューラルネットワークの登録
    model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
    #chainer.serializers.load_npz('result/out.model', model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

# イテレータの定義
    train_iter = chainer.iterators.SerialIterator(train, batchsize) #学習用
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False) #評価用

# アップデータの登録
    updater = training.StandardUpdater(train_iter, optimizer)

# トレーナの登録
    trainer = training.Trainer(updater, (epoch, 'epoch'))

# 学習状況の表示や保存
    trainer.extend(extensions.LogReport()) # ログ
    trainer.extend(extensions.Evaluator(test_iter, model)) #エポック数の表示
    trainer.extend(extensions.PrintReport(['epoch','main/loss','validation/main/loss','main/accuracy','validation/main/accuracy','elapsed_time'])) #計算状態の表示
    #trainer.extend(extensions.dump_graph('main/loss')) #ニューラルネット枠の構造
    #trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'],'epoch',file_name='loss.png')) #誤差のグラフ
    #trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'],'epoch', file_name='accuracy.png')) #精度のグラフ
    #trainer.extend(extensions.snapshot(), trigger=(100, 'epoch')) #学習再開のためのスナップショット出力
    #chainer.serializers.load_npz('result/snapshot_iter_500', trainer) #再開用
    #chainer.serializers.save_npz('result/out.model', model)

#学習開始
    trainer.run()
