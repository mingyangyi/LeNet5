import batch_norm
import LeNet5
import optimizer

import tensorflow as tf
import numpy as np

BATCH_SIZE=100#批量大小
LEARNING_RATE_BASE=0.0003#基础学习率，书中的基础学习率设置的太大了，高度非线性跳，导致梯度会很大，学习率一定要小。
REGULARIZATION_RATE=0.001#正则项损失系数
TRAINING_STEPS=10000#训练轮数
GRAD_CLIP=1.0

def average_gradients(tower_grads):

    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads=[]
        for g,_ in grad_and_vars:
            expanded_g=tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad=tf.concat(grads,0)
        grad=tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def loss(logits,labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    return cross_entropy_mean

def top_k_right(predictions, labels, k):
  '''
  Calculate the top-k error
  :param predictions: 2D tensor with shape [batch_size, num_labels]
  :param labels: 1D tensor with shape [batch_size, 1]
  :param k: int
  :return: tensor with shape [1]
  '''
  in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
  num_correct = tf.reduce_sum(in_top1)
  return num_correct

class Trainer(object):
    def __init__(self):
        self.global_step = []
        self.batch_queue = np.array([], np.int64)

        self.device_main = '/gpu:0'
        self.device_model = '/gpu:0'
        self.device_opt = '/gpu:0'

        self.decay_step = []
        self.max_step = []
        self.num_classes = []

    def placeholders(self):
        self.image_placeholders = []
        self.label_placeholders = []
        self.learn_rate_placeholder = tf.placeholder(tf.float32, shape=[1], name='learn_rate')
        self.delta_placeholder=tf.placeholder(tf.float32,shape=[1],name="delta")
        self.time_placeholder=tf.placeholder(tf.float32,shape=[1],name="times")
        self.batch_norm_param_placeholder = tf.placeholder(tf.float32, shape=[1], name='bn_eval')

    def initialize_optimizer(self,model):
        self.global_step=tf.get_variable('global_step', [],initializer=tf.zeros_initializer(), trainable=False)

        if model=="grassmann":
            opt=optimizer.Sgd_on_grassmann(self.learn_rate_placeholder,self.delta_placeholder,self.time_placeholder,grad_clip=GRAD_CLIP)

        elif model=="obilique":
            opt=optimizer.Sgd_on_obilique(self.learn_rate_placeholder,self.delta_placeholder,self.time_placeholder,grad_clip=GRAD_CLIP)

        elif model=="grassmann_with_noise":
            opt=optimizer.Sgd_on_grassmann_with_noise(self.learn_rate_placeholder,self.time_placeholder,grad_clip=GRAD_CLIP)

        elif model=="obilique_with_noise":
            opt=optimizer.Sgd_on_obilique_with_noise(self.learn_rate_placeholder,self.time_placeholder,grad_clip=GRAD_CLIP)

        else:
            assert False, 'unknown optimizer'

    def build_graph_model(self, opt,train,regularizer):
        tower_grads = []
        tower_losses = []
        tower_right = []
        tower_cross_entropy = []

        with tf.variable_scope(tf.get_variable_scope()):
            image_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,LeNet5.IMAGE_SIZE,LeNet5.IMAGE_SIZE,LeNet5.NUM_CHANNELS],name='image_placeholder')
            label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None],name='label_placeholder')

            self.image_placeholders.append(image_placeholder)
            self.label_placeholders.append(label_placeholder)

            logits=LeNet5.inference(image_placeholder,train,regularizer)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            cross_entropy_mean = loss(logits, label_placeholder)

            weight = [i for i in tf.trainable_variables() if 'weight' in i.name]
            bias = [i for i in tf.trainable_variables() if 'bias' in i.name]
            beta = [i for i in tf.trainable_variables() if 'beta' in i.name]
            gamma = [i for i in tf.trainable_variables() if 'gamma' in i.name]

            assert len(weight) + len(bias) + len(beta) + len(gamma) == len(tf.trainable_variables())

            total_loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
            t_1_right = top_k_right(logits, label_placeholder, 1)

            if opt!=None:
              # Calculate the gradients for the batch of data on this CIFAR tower.
              grads = opt.compute_gradients(total_loss)
              tower_grads.append(grads)

            tower_losses.append(total_loss)
            tower_cross_entropy.append(cross_entropy_mean)
            tower_right.append(t_1_right)

        average_grads = average_gradients(tower_grads)

        cross_entropy = tf.reduce_mean(tower_cross_entropy, name='cross_entropy')
        losses = tf.reduce_mean(tower_losses)
        right = tf.reduce_sum(tower_right)

        return average_grads, losses, right, cross_entropy




