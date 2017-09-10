import chainer
import chainer.links as L
import chainer.functions as F

class Net(chainer.Chain):
    def __init__(self, n_out):
        super(Net, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(3, 32, ksize=3, pad=1)
            self.c2 = L.Convolution2D(32, 64, ksize=3, pad=1)
            self.c3 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.fc1 = L.Linear(None, 128)
            self.fc2 = L.Linear(None, n_out)
            
            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(64)
            self.bn3 = L.BatchNormalization(64)
            
    def __call__(self, x):
        h = F.relu(self.bn1(self.c1(x)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.bn2(self.c2(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.bn3(self.c3(h)))
        h = F.relu(self.fc1(h))
        return self.fc2(h)
