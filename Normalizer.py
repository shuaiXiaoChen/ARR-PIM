import torch
import torch.nn as nn


class Detrender(nn.Module):
    def __init__(self, num_features: int, gamma=0.99):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        """
        super(Detrender, self).__init__()
        self.num_features = num_features    # 输入数据的特征数量或通道数量
        self.gamma = gamma  # 用于更新均值的参数，默认为 0.99
        # 初始化一个形状为 (1, 1, num_features) 的零张量，并将其设为不需要梯度更新的参数
        self.mean = nn.Parameter(torch.zeros(1, 1, self.num_features), requires_grad=False)

    # 根据 mode 参数的不同，调用不同的私有方法进行归一化或反归一化处理。
    def forward(self, x, mode:str):
        if mode == 'norm':
            x = self._normalize(x)  # 归一化，减去均值
        elif mode == 'denorm':
            x = self._denormalize(x)    # 反归一化，加上均值
        else: raise NotImplementedError
        return x

    # 计算 x 的均值，并更新 self.mean
    def _update_statistics(self, x):
        dim2reduce = tuple(range(0, x.ndim-1))  # 指定了需要计算均值的维度（除了最后一个维度）
        # dim2reduce 指定沿第0维和第(x.ndim-1)维计算均值，计算后的 mu 形状为 (1, 1, x.ndim)。detach() 确保 mu 不再参与梯度计算。
        mu = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.mean.lerp_(mu, 1-self.gamma)   # 以 1 - self.gamma 的速率对 self.mean 进行线性插值

    # 设置 self.mean 的值(通过手动设置)，好像用不上吧
    def _set_statistics(self, x:torch.Tensor):
        self.mean = nn.Parameter(x, requires_grad=False)

    # 对输入 x 进行归一化处理，即减去均值
    def _normalize(self, x):
        x = x - self.mean
        return x

    # 对输入 x 进行反归一化处理，即加上均值
    def _denormalize(self, x):
        x = x + self.mean
        return x