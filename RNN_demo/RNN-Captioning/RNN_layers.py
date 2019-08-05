import numpy as np
import RNN_demo

def rnn_step_forward(X, pre_h, Wx, Wh, b):
    """
    rnn 正向传播 next_h = tanh(X*Wx + pre_h*Wh + b)
    :param X: 数据集，(N, D)
    :param pre_h: 上一个隐藏状态，(N, H)
    :param Wx: X对应的权重矩阵，(D, H)
    :param Wh: 隐藏状态对应的权重矩阵，(H, H)
    :param b: 偏置项，(H, )
    :return:(next_h, cache)
            next_h: 下一个隐藏状态， (N, H)
            cache: 保存反向传播时用到的变量
    """
    temp1 = np.dot(X, Wx)
    temp2 = np.dot(pre_h, Wh)
    cache = (X, pre_h, Wx, Wh, temp1 + temp2 +b)
    next_h = np.tanh(temp1 + temp2 + b)
    return next_h

def rnn_cell_forward(xt, a_prev, parameters):
    """
    RNN单步前向传播
    :param xt:第t个时间步输入的数据，（n_x, m）
    :param a_prev:第 t - 1 个时间步的隐藏状态，（n_a, m）
    :param parameters:权重字典
                    Wax：xt乘以的权重，（n_a, n_x）
                    Waa: a_prev乘以的权重，（n_a, n_a）
                    Wya: a_next乘以的权重，输出预测信息， (n_y, n_a)
                    ba: Waa对应的偏置项，（n_a, ）
                    by: Wya对应的偏置项，（n_y, ）
    :return:(a_next, yt_pred, cache)
            cache:(a_next, a_pred, xt, parameters)
    """

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)


