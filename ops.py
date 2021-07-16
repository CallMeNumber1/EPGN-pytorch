# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
def relu(x):
    # return tf.nn.relu(x)
    return nn.functional.relu(x)


def discriminator_loss(loss_func, real, fake):
    loss = 0
    # if loss_func.__contains__('bgan'):
    #     loss = tf.reduce_mean(fake-real)

    # if loss_func == 'lsgan':
    #     loss = tf.reduce_mean(tf.square(fake)) - tf.reduce_mean(tf.squared_difference(real,1.0))

    # if loss_func == 'hinge':
    #     loss = tf.reduce_mean(relu(1.0+fake))+ tf.reduce_mean(relu(1.0 -real))

    if loss_func == 'wgan':
        loss = -torch.mean(torch.log(real) + torch.log(1-fake))
        # loss = -tf.reduce_mean(tf.log(real) + tf.log(1-fake))

    return loss

def generator_loss(loss_func, real, fake):
    loss = 0
    # if loss_func.__contains__('bgan'):
    #     loss = tf.reduce_mean(real-fake)

    # if loss_func == 'lsgan':
    #     loss = tf.reduce_mean(tf.squared_difference(fake,1.0))

    # if loss_func == 'hinge':
    #     loss = -tf.reduce_mean(fake)

    if loss_func == 'wgan':
        # loss = -tf.reduce_mean(tf.log(fake))
        loss = -torch.mean(torch.log(fake))
    return loss


def euclidean_distance(a,b):
    # a.shape = N x D
    # b.shape = M x D
    # N, D = tf.shape(a)[0], tf.shape(a)[1]
    # M = tf.shape(b)[0]
    # a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    # b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    # return tf.reduce_mean(tf.square(a-b), axis=2)
    '''
        由于放到gpu上计算太消耗显存，放到cpu上tensor的话又太慢，所以尝试用numpy计算
        事实证明不太行，要使用numpy计算就要先detach()，这样梯度无法反传
        最终尝试循环计算，每次计算一个样本和所有语义属性在视觉原型的距离。
        N:bs D:2048 M:40
    '''
    # N, D = a.shape[0], a.shape[1]
    # M = b.shape[0]
    # a = np.tile(np.expand_dims(a, 1), (1,M,1))
    # b = np.tile(np.expand_dims(b, 0), (N,1,1))
    # 循环，分别并行
    # return np.mean((a - b)**2, axis=2)
    N, D = a.shape[0], a.shape[1]
    M = b.shape[0]
    dists = torch.zeros(N, M).cuda()
    for i in range(N):
        row = a[i,:].repeat(M, 1) 
        temp = torch.mean((row - b)**2, dim=1).view(-1, M)
        dists[i] = temp
    # 全部并行
    # N, D = a.shape[0], a.shape[1]
    # M = b.shape[0]
    # a = a.unsqueeze(1).repeat(1,M,1)
    # b = b.unsqueeze(0).repeat(N,1,1)
    # dists = torch.mean((a - b)**2, dim=2)
    return dists


# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.01)
#     return tf.Variable(initial)

# def bias_variable(shape):
#     initial = tf.constant(0.01, shape=shape)
#     return tf.Variable(initial)

# def dense(x, in_dim, out_dim):
#     weights = weight_variable([in_dim, out_dim])
#     bias = bias_variable([out_dim])
#     out = tf.add(tf.matmul(x, weights), bias)
#     return out



