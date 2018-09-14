import tensorflow as tf
from layers import _variable_on_device

def batch_norm(inputs,scale=None,epsilon=1e-5,name="batch_norm",device=None):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        #beta=_variable_on_device("beta",shape=[inputs.get_shape()[0]],
                                 #initializer=tf.zeros_initializer(),
                                 #trainable=True,
                                 #device=device)
        #if scale==True:
            #gamma=_variable_on_device("gamma",shape=[inputs.get_shape()[0]],
                                 #initializer=tf.ones_initializer(),
                                 #trainable=True,
                                 #device=device)
        #else:
            #gamma=None

        #reduced_dim=[i for i in range(len(inputs.get_shape())-1)]
        #print(inputs)
        batch_mean,batch_var=tf.nn.moments(inputs,axes=[0,1,2],keep_dims=False)
    return (inputs-batch_mean)/(batch_var+1e-5)
 #tf.nn.batch_normalization(inputs,batch_mean,batch_var,offset=None,scale=None,variance_epsilon=epsilon)



