import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops

def norm(v):
    v = tf.reshape(v, shape=[-1,1])
    return tf.sqrt(tf.matmul(tf.transpose(v),v)[0][0])

def unit(v,eps=1e-8):
    x=tf.reshape(v,shape=[-1,1])
    vnorm=norm(x)
    return v/(vnorm+eps)

def xTy(x,y):
    dim=x.get_shape()[0]
    x=tf.reshape(x,[dim,1])
    y=tf.reshape(y,[dim,1])
    return tf.matmul(tf.transpose(x),y)[0,0]

def clip_by_norm(x,clip_norm):
    tf.convert_to_tensor(clip_norm,tf.float32)
    n=norm(x)+1e-8
    return unit(x)*tf.minimum(n,clip_norm)

def grassmann_project(x,eta):
    n=norm(x)
    return eta-xTy(x,eta)*x/tf.square(n)

def grassmann_retrction(x,eta):
    return unit(x)*tf.cos(norm(eta))+unit(eta)*tf.sin(norm(eta))

def oblique_project(x,eta):
    ddiag=xTy(x,eta)
    p=eta-x*ddiag
    return p

def oblique_retrction(x,eta):
    v=x+eta
    return unit(v,1e-8)


class unit_initializer(init_ops.Initializer):
    def __init__(self, seed=None, dtype=tf.float32, eps=1e-8):
        self.seed = seed
        self.dtype = dtype
        self.eps = eps

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        v = random_ops.truncated_normal(shape, 0, 1.0, dtype, seed=self.seed)
        return unit(v,self.eps)
