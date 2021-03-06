# RGB-BGR識別

近年、Deep Learningの登場により画像認識の精度が大幅に向上した。その用途は、クラス識別だけではなく、
検出やセマンティックセグメンテーションなど多岐に渡っている。しかしながら、性能が大幅に向上した現在においても、
入力画像の色の順序（RGB/BGR)を間違えただけで、精度が大幅に低下してしまう。開発者たちは
日常的に、この色の順序の問題と戦っており、問題発見およびデバッグに多くの時間を裂いている。本研究では
この解決に向け、色の順序を認識する手法を開発した。

## 使い方

requirements: chainer >= 2.0, numpy, pillow。

### 学習

```
python train.py --gpu 0
```

### 識別

```
python predict.py -i cat.jpg
```
