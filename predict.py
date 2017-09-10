from __future__ import print_function
import argparse
import numpy as np

import chainer
from PIL import Image

from net import *
    
def gen_dataset(data):
    image_rgb = data.copy()
    image_bgr = data[:,::-1,:,:]
    labels_rgb = np.zeros((len(data),), np.int32)
    labels_bgr = np.ones((len(data),), np.int32)
    images = np.concatenate((image_rgb, image_bgr), axis=0)
    labels = np.concatenate((labels_rgb, labels_bgr), axis=0)

    return chainer.datasets.tuple_dataset.TupleDataset(images, labels)

def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--image', '-i', help='Input image', required=True)
    parser.add_argument('--model', '-m', default='./result/net_epoch_30',
                        help='trained model')
    args = parser.parse_args()

    net = Net(2)
    chainer.serializers.load_npz(args.model, net)
    
    # input data
    img = Image.open(args.image)
    img = img.resize((32, 32))
    img = np.asarray(img)
    img = img.astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))
    
    x = chainer.Variable(img[np.newaxis,:,:,:])
    
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            y = F.softmax(net(x))
        
    if y.data[0,0] > y.data[0,1]:
        print('RGB')
    else:
        print('BGR')        

if __name__ == '__main__':
    main()