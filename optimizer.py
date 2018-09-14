import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

import gutils

class Sgd_on_grassmann(optimizer.Optimizer):

    def __init__(self,learning_rate,delta,times,grad_clip=None,use_locking=False,name="Sgd_on_grassmann"):
        super(Sgd_on_grassmann, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._grad_clip = grad_clip
        self._delta=delta
        self._times=times

         # Tensor versions of the constructor arguments, created in _prepare().
        self._learning_rate_t = None
        self._delta_t=None
        self._grad_clip_t = None
        self._times_t=None

    def _prepare(self):
        self._learning_rate_t = tf.convert_to_tensor(self._learning_rate, name="learning_rate")
        self._delta_t = tf.convert_to_tensor(self._delta, name="delta")
        self._times_t=tf.convert_to_tensor(self._times,name="times")
        if self._grad_clip != None:
            self._grad_clip_t = tf.convert_to_tensor(self._grad_clip, name="grad_clip")
        else:
            self._grad_clip_t=None

    def _apply_dense_on_grasssmann(self,grad_on_grassmann,grad_on_obilique,var):
        a=tf.maximum(self._delta_t,1/(tf.square(self._times)))
        b_1=2*(1-a)*tf.matmul(tf.transpose(grad_on_grassmann),gutils.grassmann_project(var,grad_on_obilique))
        b_2=gutils.norm(gutils.grassmann_project(grad_on_obilique))
        b=b_1/b_2

        if self._grad_clip !=None:
            h=self._learning_rate_t*(a*grad_on_grassmann+b*gutils.grassmann_project(var,grad_on_obilique))
            h=-gutils.clip_by_norm(h,self._grad_clip_t)
        else:
            h = -self._learning_rate_t * (a * grad_on_grassmann + b * gutils.grassmann_project(var, grad_on_obilique))

        var_update=gutils.grassmann_retrction(var,h)
        return var_update

class Sgd_on_obilique(optimizer.Optimizer):

    def __init__(self,learning_rate,delta,times,grad_clip=None,use_locking=False,name="Sgd_on_grassmann"):
        super(Sgd_on_grassmann, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._grad_clip = grad_clip
        self._delta=delta
        self._times=times

         # Tensor versions of the constructor arguments, created in _prepare().
        self._learning_rate_t = None
        self._delta_t=None
        self._grad_clip_t = None
        self._times_t=None

    def _prepare(self):
        self._learning_rate_t = tf.convert_to_tensor(self._learning_rate, name="learning_rate")
        self._delta_t = tf.convert_to_tensor(self._delta, name="delta")
        self._times_t=tf.convert_to_tensor(self._times,name="times")
        if self._grad_clip != None:
            self._grad_clip_t = tf.convert_to_tensor(self._grad_clip, name="grad_clip")
        else:
            self._grad_clip_t=None

    def _apply_dense_on_obilique(self,grad_on_grassmann,grad_on_obilique,var):
        a=tf.maximum(self._delta_t,1/(tf.square(self._times)))
        b_1=2*(1-a)*tf.matmul(tf.transpose(grad_on_obilique),gutils.obilique_project(var,grad_on_grassmann))
        b_2=gutils.norm(gutils.obilique_project(grad_on_grassmann))
        b=b_1/b_2

        if self._grad_clip !=None:
            h=self._learning_rate_t*(a*grad_on_obilique+b*gutils.obilique_project(var,grad_on_grassmann))
            h=gutils.clip_by_norm(h,self._grad_clip_t)
        else:
            h = -self._learning_rate_t * (a * grad_on_obilique + b * gutils.obilique_project(var, grad_on_grassmann))

        var_update=gutils.obilique_retrction(var,h)
        return var_update

class Sgd_on_grassmann_with_noise(optimizer.Optimizer):

    def __init__(self,learning_rate,times,grad_clip=None,use_locking=False,name="Sgd_on_grassmann_with_noise"):
        super(Sgd_on_grassmann, self).__init__(use_locking, name)
        self._learning_rate=learning_rate
        self._times=times
        self._grad_clip=grad_clip

        self._learning_rate_t = None
        self._delta_t = None
        self._grad_clip_t = None
        self._times_t = None

    def _prepare(self):
        self._learning_rate_t = tf.convert_to_tensor(self._learning_rate, name="learning_rate")
        self._times_t=tf.convert_to_tensor(self._times,name="times")
        if self._grad_clip != None:
            self._grad_clip_t = tf.convert_to_tensor(self._grad_clip, name="grad_clip")
        else:
            self._grad_clip_t=None

    def _apply_dense_on_grassmann_with_noise(self, grad, var,seed):
        g=gutils.grassmann_project(var,grad)
        g_norm=gutils.norm(g)
        if g_norm>=1/(self._times):
            a=1-1/(tf.square(self._times)*tf.square(g_norm))
        else:
            a=1/tf.square(self._times)
        b=1/tf.square(self._times)

        dim=grad.get_shape()[0]
        noise=tf.truncated_normal([dim,dim],mean=0.0,stddev=1.0,dtype=tf.float32,seed=seed,name="random_noise")

        if self._grad_clip==None:
            h=-self._learning_rate_t*(a*g+b*noise)
        else:
            h = -self._learning_rate_t * (a * g + b * noise)
            h=gutils.clip_by_norm(h,self._grad_clip_t)

        var_new=gutils.grassmann_retrction(var,h)

        return var_new


class Sgd_on_obilique_with_noise(optimizer.Optimizer):

    def __init__(self, learning_rate, times, grad_clip=None, use_locking=False, name="Sgd_on_obilique_with_noise"):
        super(Sgd_on_obilique, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._times = times
        self._grad_clip = grad_clip

        self._learning_rate_t = None
        self._delta_t = None
        self._grad_clip_t = None
        self._times_t = None

    def _prepare(self):
        self._learning_rate_t = tf.convert_to_tensor(self._learning_rate, name="learning_rate")
        self._times_t = tf.convert_to_tensor(self._times, name="times")
        if self._grad_clip != None:
            self._grad_clip_t = tf.convert_to_tensor(self._grad_clip, name="grad_clip")
        else:
            self._grad_clip_t = None

    def _apply_dense_on_obilique_with_noise(self, grad, var, seed):
        g = gutils.obilique_project(var, grad)
        g_norm = gutils.norm(g)
        if g_norm >= 1 / (self._times):
            a = 1 - 1 / (tf.square(self._times) * tf.square(g_norm))
        else:
            a = 1 / tf.square(self._times)
        b = 1 / tf.square(self._times)

        dim = grad.get_shape()[0]
        noise = tf.truncated_normal([dim, dim], mean=0.0, stddev=1.0, dtype=tf.float32, seed=seed, name="random_noise")

        if self._grad_clip==None:
            h = -self._learning_rate_t * (a * g + b * noise)
        else:
            h = -self._learning_rate_t * (a * g + b * noise)
            h = gutils.clip_by_norm(h, self._grad_clip_t)

        var_new = gutils.grassmann_retrction(var, h)

        return var_new






