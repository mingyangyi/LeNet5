import tensorflow as tf
import gutils

def apply_dense_on_grasssmann(grad_clip, grad_on_grassmann, grad_on_oblique, var,learning_rate,times,delta):
    a = tf.maximum(delta, 1 / tf.log((tf.log((times+2)))))
    n = gutils.unit(gutils.grassmann_project(var, grad_on_oblique)) * gutils.norm(grad_on_grassmann)
    b_1 = 2 * (1 - a) * gutils.xTy(grad_on_grassmann, n)
    b_2 = gutils.norm(grad_on_grassmann)
    b = b_1 / (b_2+1e-5)

    if grad_clip != None:
        h = learning_rate * (a * grad_on_grassmann + b * n)
        h = -1*gutils.clip_by_norm(h, grad_clip)
    else:
        h = -1*learning_rate * (a * grad_on_grassmann + b * n)

    var_update = gutils.grassmann_retrction(var, h)
    return var_update

def apply_dense_on_grasssmann_g(grad_clip, grad_on_grassmann, var,learning_rate,times,delta):
    a = tf.maximum(delta, 1)#/ (tf.log(times+2))

    if grad_clip != None:
        h = learning_rate * (a * grad_on_grassmann )
        h = -1*gutils.clip_by_norm(h, grad_clip)
    else:
        h = -1*learning_rate * a * grad_on_grassmann

    var_update = gutils.grassmann_retrction(var, h)
    return var_update

def _apply_dense_on_oblique(grad_clip, grad_on_grassmann, grad_on_oblique, var,learning_rate,times,delta):
    a = tf.maximum(delta, 1 / tf.log((tf.log((times + 2)))))
    n = gutils.unit(gutils.oblique_project(var, grad_on_grassmann))*gutils.norm(grad_on_oblique)
    b_1 = 2 * (1 - a) * gutils.xTy(grad_on_oblique, n)
    b_2 = gutils.norm(grad_on_oblique)
    b = b_1 / (b_2 + 1e-5)

    if grad_clip !=None:
        h = -1 * learning_rate * (a * grad_on_oblique + b * n)
        h = gutils.clip_by_norm(h, grad_clip)
    else:
        h = -1*learning_rate * (a * grad_on_oblique + b * n)

    var_update=gutils.oblique_retrction(var,h)
    return var_update

def _apply_dense_on_oblique_o(grad_clip, grad_on_oblique, var,learning_rate,times,delta):
    a=tf.maximum(delta,1)#/(tf.log(times+2))

    if grad_clip !=None:
            h=-1*learning_rate*(a*grad_on_oblique)
            h=gutils.clip_by_norm(h,grad_clip)
    else:
            h = -1*learning_rate * (a * grad_on_oblique )

    var_update=gutils.oblique_retrction(var,h)
    return var_update

def _apply_dense_on_grassmann_with_noise(grad_clip,grad, var,seed,learning_rate,times):
    g=gutils.grassmann_project(var,grad)
    g_norm=gutils.norm(g)

    #a=tf.minimum(1-1/(tf.square(times+1)*tf.square(g_norm)+1e-5),1/tf.square(times+1))
    a=1.0

    b=1/tf.square(times+1)

    dim=tf.convert_to_tensor(grad.get_shape()[0],dtype=tf.int32)

    noise=tf.truncated_normal([dim,1],mean=0.0,stddev=0.0001,dtype=tf.float32,seed=seed,name="random_noise")

    if grad_clip==None:
        h=-learning_rate*(a*g+b*noise)
    else:
        h = -learning_rate * (a * g + b * noise)
        h=gutils.clip_by_norm(h,grad_clip)

    var_new=gutils.grassmann_retrction(var,h)
    return var_new

def _apply_dense_on_oblique_with_noise(grad_clip,grad, var,seed,learning_rate,times):
    g = gutils.oblique_project(var, grad)
    g_norm = gutils.norm(g)
    #a = tf.minimum(1 - 1 / (tf.square(times + 1) * tf.square(g_norm) + 1e-5), 1 / tf.square(times + 1))
    a=1.0
    b = 1 / (tf.square(times+1))

    dim = tf.convert_to_tensor(grad.get_shape()[0],dtype=tf.int32)
    noise = tf.truncated_normal([dim, 1], mean=0.0, stddev=0.0001, dtype=tf.float32, seed=seed, name="random_noise")

    if grad_clip==None:
        h = -1*learning_rate * (a * g + b * noise)
    else:
        h = -1*learning_rate * (a * g + b * noise)
        h = gutils.clip_by_norm(h, grad_clip)

    var_new = gutils.grassmann_retrction(var, h)

    return var_new