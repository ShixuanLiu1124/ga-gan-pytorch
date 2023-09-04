import numpy as np
import torch
import time



def count_time(func):
    """
    Statistical function runtime decorator
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('Function：<' + str(func.__name__) + '>TimeCost：{:.2f} Minute'.format((end_time - start_time) / 60))
        return ret

    return wrapper


def xavier_init(size):  # 初始化参数时使用的xavier_init函数
    in_dim = size[0]
    xavier_stddev = 1.0 / torch.sqrt(in_dim / 2.0)  # 初始化标准差
    return torch.randn(size) * xavier_stddev


def sample_Z(m, n):  # 生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.uniform(-1, 1., size=[m, n])
