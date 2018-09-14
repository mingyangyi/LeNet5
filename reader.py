import os
import pickle
import numpy as np

file="D:/tmp/cifar-10-python/cifar-10-batches-py"

def unpickle(file):
    data,labels=[],np.zeros(shape=[50000,10])
    for i in range(5):
        file_dir = os.path.join(file, 'data_batch_{}'.format(i+1))
        with open(file_dir, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x = [[float(v) for v in u] for u in dict[b'data']]
            v=0
            for u in dict[b'labels']:
                labels[(i*10000+v), u]=1.0
                v=v+1
            data.extend(x)
    return data,labels

def test_unpickle(file):
    data, labels = [], np.zeros([10000,10])
    file_dir = os.path.join(file, 'test_batch')
    with open(file_dir, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x = [[float(v) for v in u] for u in dict[b'data']]
        v = 0
        for u in dict[b'labels']:
            labels[v, u] = 1.0
            v = v + 1
        data.extend(x)
    return data, labels
