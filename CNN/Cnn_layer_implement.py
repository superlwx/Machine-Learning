import numpy as np


"""

实现卷积神经网络中卷积层和池化层

"""


def conv_forward_naive(x, w, b, conv_param):
    """
    一个卷积层前向传播的简单实现
    :param x:输入数据，（N, C, H, W）
    :param w:卷积核的权重， (F, C, HH, WW)
    :param b:偏置项， (F, )
    :param conv_param:一个参数字典：
        - stride :卷积核的步长
        - pad :使用零填充的大小
    :return out:输出计算后的数据，(N, F, H', W')
            H' = (H + 2 * pad - HH) / stride
            W' = (W + 2 * pad - WW) / stride
    :return cache:保存输入的数据,(x, w, b, conv_param)
    """
    stride = conv_param['stride']
    pad = conv_param['stride']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    H_out = int((H + 2 * pad - HH) / stride) + 1  # 对应H'
    W_out = int((W + 2 * pad - WW) / stride) + 1  # 对应W'
    out = np.zeros((N, F, H_out, W_out))
    for n in range(N):
        for f in range(F):
            new_matrix = np.ones((H_out, W_out)) * b[f]  # 加入偏置项
            for c in range(C):
                padded_x = np.lib.pad(x[n, c], pad_width=pad,mode="constant", constant_values=0)  # 填充矩阵
                for i in range(H_out):
                    for j in range(W_out):
                        new_matrix[i, j] += np.sum(padded_x[stride * i: stride * i + HH,
                                                  stride * j: stride * j + WW] * w[f, c, :, :])
                out[n, f] = new_matrix
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    一个卷积层反向传播的简单实现
    :param dout:上游梯度 维度与out一致，(N, F, H', W')
    :param cache:(x, w, b,conv_param)
    :return:dx, dw,db:各个参数的梯度，维度与x, w, b一致
    """
    (x, w, b, conv_param) = cache
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    (N, F, H_out, W_out) = dout
    padded_x = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                          mode='constant', constant_values=0)  # 在第3，第4个维度上填充
    padded_dx = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    db[f] += dout[n, f, i, j]
                    dw[f] += padded_x[n, :, stride * i: stride * i + HH,
                             stride * j: stride * j + WW] * dout[n, f, i, j]
                    padded_dx[n, :, stride * i: stride * i + HH,
                             stride * j: stride * j + WW] += w[f] * dout[n, f, i, j]
    dx = padded_dx[:, :, pad: pad + H, pad: pad + W]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):

    """
    一个最大池化层前向传播的简单实现
    :param x: 输入的数据，维度为（N, C, H, W）
    :param pool_param: 一个参数字典
        - pool_width:宽度
        - pool_height:高度
        - stride:步长
    :return:
        out:输出数据，维度为（N, C, H', W'）
        cache:缓存输入数据，(x, pool_param)
    """

    (N, C, H, W) = x.shape
    pool_height = pool_param['height']
    pool_width = pool_param['width']
    stride = pool_param['stride']
    H_out = int((H - pool_height) / stride) + 1
    W_out = int((W - pool_width) / stride) + 1
    out = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(c):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, c, i, j] = np.sum(x[n, c, stride * i: stride * i + pool_height,
                                             stride * j: stride * j + pool_width])
    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    """

    一个最大池化层反向传播的简单实现
    :param dout: 上游梯度，维度与out一致，(N, F, H', W')
    :param cache: （x, pool_param）
    :return:
        dx: 参数x的梯度，维度与x一致，(N, C, H, W)
    """

    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    pool_height = pool_param['height']
    pool_width = pool_param['width']
    stride = pool_param['stride']
    H_out = int((H - pool_height) / stride) + 1
    W_out = int((W - pool_width) / stride) + 1
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    window = x[n, c, stride * i:stride * i + pool_height, stride * j: stride * j + pool_width]
                    dx[n, c] = (window == np.sum(window)) * dout[n, c, i, j]
    return dx



