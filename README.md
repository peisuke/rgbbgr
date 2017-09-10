# RGB-BGR識別

近年、Deep Learningの登場により画像認識の精度が大幅に上がってきた。その用途は、クラス識別だけではなく、
検出やセマンティックセグメンテーションなど多岐に渡っている。しかしながら、性能が大幅に向上した現在においても、
入力画像の色の順序（RGB/BGR)を間違えただけで、精度が大幅な低下してしまう。自動運転を始めとし、我々の生活の
中にAIが日常的に用いられる中で、ソフトウェアのバグは最悪の場合死に至ることもある重大な問題である。開発者たちは
日常的に、この色の順序の問題と戦っており、問題発見およびデバッグに多くの時間を裂いている。本研究では
この解決に向け、色の順序を認識する手法を開発した。

## 使い方

- ソース読め