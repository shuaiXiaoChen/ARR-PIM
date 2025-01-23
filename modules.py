import torch
import torch.nn as nn
import torch.nn.functional as F
from Normalizer import Detrender

import math
from math import sqrt
from einops import rearrange, reduce, repeat


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) # 调换维度数据，符合卷积需要(batch, features, time_steps)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back ，返回原始维度(batch, time_steps, features)


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim


        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x) # 为什么只提取最后一个时间步呢？
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

        # -------------新增的趋势感知模块---------- #
        normalization = "None"  # 手动控制，默认为None
        gamma = 0.99
        num_channels = in_dim
        self.normalization = normalization

        if self.normalization == "Detrend":
            self.use_normalizer = True
            self.normalizer = Detrender(num_channels, gamma=gamma)
        else:
            self.use_normalizer = False
        # -------------新增的趋势感知模块---------- #


    def forward(self, x):
        x1 = x
        if self.use_normalizer:
            x1 = self.normalizer(x1, "norm")  # 进行归一化处理
        # x will be last hidden state of the GRU layer
        h_end = x1
        # h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        # h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        if self.use_normalizer:
            decoder_out = self.normalizer(decoder_out, "denorm")    # 反归一化处理这里的x1应当是隐层输出
        out = self.fc(decoder_out)
        return out

class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)] # 这是第一层
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim)) # 这是其他层

        layers.append(nn.Linear(hid_dim, out_dim)) # 这是最后一层，一共有n层

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) # 防止过拟合
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1): # 每一层都使用droupout技术，因为for
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


# -----------------------modules from MAGNN---------------------------#
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A): # (x, A)：输入张量 x 和邻接矩阵 A。
        """
            'ncvl,vw->ncwl'：定义了张量乘积的模式。
            n：批次维度
            c：特征维度
            v：节点维度
            l：序列长度
            x 的形状为 (batch_size, num_channels, num_vertices, seq_length)。
            A 的形状为 (num_vertices, num_vertices)。
            输出张量的形状为 (batch_size, num_channels, num_vertices, seq_length)。
            x 的维度 v（节点维度）与 A 的维度 v 相乘。
            结果是每个节点的特征根据邻接矩阵 A 进行加权求和，从而实现节点特征在图结构上的传播。
        """
        # print("x.shape:",x.shape,"A.shape:",A.shape)
        # x = torch.einsum('ncvl,vw->ncwl',(x,A)) # 爱因斯坦求和约定的实现，用于简洁地表示复杂的张量操作
        x = torch.einsum('bwn,nm->bwm',(x,A)) # 爱因斯坦求和约定的实现，用于简洁地表示复杂的张量操作
        # print("x(after torch.einsum).shape",x.shape)
        return x.contiguous() # 保持内存连续性

    """
        这个 linear 类通过 1x1 卷积层实现了线性变换，可以高效地处理多维度数据，
        尤其适用于图神经网络和其他复杂结构的数据。它的使用方式与传统的全连接层相似，
        但在处理多维张量时具有更大的灵活性和效率。
    """

class linear(nn.Module):  # 定义初始化函数，接受三个参数：输入通道数 c_in、输出通道数 c_out 和一个布尔值 bias，表示是否使用偏置。
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        # print("-----class linear(nn.Module)-----")
        # print("class linear(nn.Module):\n",x.shape)
        return self.mlp(x)


class LinearForReshape(nn.Module):
    def __init__(self):
        super(LinearForReshape, self).__init__()
        self.linear_layer = nn.Linear(56*38, 100*38)

    def forward(self, x):
        # 调整输入张量的形状
        # print("x.size():",x.size())
        batch_size, channels, length = x.size()
        x = x.permute(1, 0, 2).reshape(channels, -1)
        x = self.linear_layer(x)
        x = torch.relu(x)
        # print("x.size():",x.size())
        x = x.view(channels, 100, length)  # 重新调整为目标形状
        return x


class Conv1DReshape(nn.Module):
    def __init__(self):
        super(Conv1DReshape, self).__init__()
        # 定义1D卷积层，将输入通道数从56变为100
        self.conv1d = nn.Conv1d(in_channels=56, out_channels=100, kernel_size=1)

    def forward(self, x):
        # 输入张量的形状 (56, 256, 38)
        # print(f"Input shape: {x.shape}")  # 调试信息

        if len(x.size()) != 3:
            raise ValueError("Input tensor must be 3-dimensional")

        # 调整输入张量的形状为 (256, 56, 38)，以便应用1D卷积
        x = x.permute(1, 0, 2)  # (256, 56, 38)

        # 通过1D卷积层处理，调整为 (256, 100, 38)
        x = self.conv1d(x)

        # 最终输出形状为 (256, 100, 38)
        # print(f"Output shape: {x.shape}")  # 调试信息

        return x

class mixprop(nn.Module):  # 多层图卷积
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()    # 输入input与邻接矩阵进行张量运算
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep  # 存储传播的阶数。
        self.dropout = dropout
        self.alpha = alpha  # 存储残差连接的权重参数。
        """
            c_in：输入特征的通道数。
            x 是初始输入特征张量，形状为 (batch_size, num_nodes, c_in)
            c_out：输出特征的通道数。
            gdep：传播的阶数，图卷积的层数。
            dropout：丢弃率，用于防止过拟合。
            alpha：残差连接中的权重参数，用于控制输入特征与传播特征的加权和。
            self.nconv：图卷积操作层。
            self.mlp：一个全连接层（线性变换），将多阶传播后的特征进行线性组合。
        """
    def forward(self, x, adj):
        # print("------class mixprop(nn.Module): --------")
        # print("x.shape:",x.shape)
        # print("adj:",adj.shape)
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # torch.eye(adj.size(0)) 生成一个单位矩阵（对角线上全为 1），这样每个节点都会在邻接矩阵中多一条到自己的边。
        d = adj.sum(1)  # 计算每个节点的度，即每个节点的邻居数量。对于每一行，将所有元素相加，得到节点度的向量 d。
        h = x  # 初始化特征 h 为输入特征 x，并将其添加到列表 out 中，以便后续拼接。
        out = [h]
        a = adj / d.view(-1, 1)  # 归一化邻接矩阵，使得邻接矩阵的每一行之和为1。（首先将邻接矩阵变成一列，然后同时除以这个和，得到归一化的每一行和为1的邻接矩阵）
        for i in range(self.gdep):
            # print("i of gdep:",i)
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)  # 表示对 h 进行图卷积操作，使用归一化邻接矩阵 a（对输入特征和邻接矩阵进行图卷积运算）。
            out.append(h)  # 在将更新后的特征添加到列表 out 中。为什么要计算两个重复的结果呢
        ho = torch.cat(out, dim=1)  # 将所有层的特征在特征维度上（即列方向）拼接，形成新的特征矩阵 ho。
        # print("to mlp next!!!!")
        ho = torch.unsqueeze(ho,0)
        # print("ho.shape:",ho.shape)
        ho = torch.permute(ho,(0,2,1,3))
        # print("ho.shape(after permute):",ho.shape)
        ho = self.mlp(ho)  # 对拼接后的特征进行线性变换，得到最终的输出特征。
        return ho


"""
    总的来说，这个类通过学习节点嵌入，然后利用这些嵌入来创建每个节点的前 k 个邻居的邻接矩阵，
    这个过程在指定的层数内重复。
    这些邻接矩阵可以用于后续的图神经网络操作，如消息传递和特征聚合。每个邻接矩阵反映了不同层次的节点关系和连接强度。
    adj_set 是一个列表，包含多个邻接矩阵，每个邻接矩阵对应于一层图构造
    这些邻接矩阵是稀疏的，保留了节点之间的前 k 个最强连接
    adj_set 的长度等于 self.layers，即图构造层的数量。
    第一个代码版本适用于需要稀疏化邻接矩阵的场景，特别是在大规模图上可以有效减少计算和存储开销。
"""
class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device,
                 alpha=3):  # 图中节点数量，n：需考虑邻居数量，dim：节点嵌入维度，layer_num：神经网络的层数，alpha：缩放因子（默认值为3）
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes  # 将节点数量保存为实例变量
        self.layers = layer_num

        self.emb1 = nn.Embedding(nnodes, dim)  # 分别定义两个嵌入层 emb1 和 emb2，用于表示节点特征，维度为 dim。
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()  # 初始化两个 ModuleList 对象，用于存储线性层。
        self.lin2 = nn.ModuleList()
        for i in range(layer_num):  # 遍历 layer_num 的范围，以创建指定数量的层。
            self.lin1.append(nn.Linear(dim, dim))  # lin1 和 lin2 中添加线性层，每个线性层将 dim 维的输入转换为 dim 维的输出。
            self.lin2.append(nn.Linear(dim, dim))

        self.device = device
        self.k = k  # 将邻居数量保存为实例变量。====子图尺寸
        self.dim = dim
        self.alpha = alpha

    def forward(self, idx, scale_idx, scale_set):  # idx：节点的索引，scale_idx：缩放索引，scale_set：每一层的缩放因子集合。
        # print("self.device(of class graph_constructor in modules.py):",self.device)
        # print("idx.device(of class graph_constructor in modules.py):",idx.device)
        nodevec1 = self.emb1(idx)  # 从 emb1 和 emb2 中获取由 idx 索引的节点嵌入。形成n行dim列的嵌入向量
        nodevec2 = self.emb2(idx)

        adj_set = []  # 初始化一个空列表，用于存储每一层的邻接矩阵。

        for i in range(self.layers):  # 遍历 self.layers 的范围，对每一层执行操作。
            nodevec1 = torch.tanh(
                self.alpha * self.lin1[i](nodevec1 * scale_set[i]))  # 对当前层的 nodevec1 和 nodevec2 应用线性变换，然后进行缩放和双曲正切激活。
            nodevec2 = torch.tanh(self.alpha * self.lin2[i](nodevec2 * scale_set[i]))
            a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1,
                                                                                                     0))  # 计算 nodevec1 和 nodevec2 及其转置的矩阵乘积之差(相似度矩阵)，结果是一个反对称矩阵 a。
            adj0 = F.relu(torch.tanh(self.alpha * a))  # 对 相似度矩阵 a 进行非线性处理，得到adj0. 应用缩放的双曲正切激活，然后应用ReLU激活，得到非负邻接矩阵 adj0。

            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)  # 指定设备上初始化一个形状为 (节点数量, 节点数量) 的零矩阵 mask。
            mask.fill_(float('0'))  # 将 mask 填充为全零（这一步是多余的，因为初始化时已经是零）。
            s1, t1 = adj0.topk(self.k, 1)  # 从 adj0 中选取每行的前 k 个最大值及其索引。
            mask.scatter_(1, t1, s1.fill_(1))  # 根据索引 t1 更新 mask，将每行的前 k 个位置设为1。
            # print(mask)
            adj = adj0 * mask  # 将 adj0 和 mask 进行逐元素相乘，仅保留每行的前 k 个值。构建稀疏邻接矩阵
            adj_set.append(adj)  # 将得到的邻接矩阵 adj 添加到 adj_set 列表中。

        return adj_set

# contrastive study
# 1.form embed.py
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer  # DACStructure（一共有3层）

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list

# model from atten.py
class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05,
                 output_attention=False):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index,
                attn_mask):

        # Patch-wise Representation
        B, L, H, E = queries_patch_size.shape  # batch_size*channel, patch_num, n_head, d_model/n_head
        scale_patch_size = self.scale or 1. / sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size,
                                         keys_patch_size)  # batch*ch, nheads, p_num, p_num
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))  # B*D_model H N N

        # In-patch Representation
        B, L, H, E = queries_patch_num.shape  # batch_size*channel, patch_size, n_head, d_model/n_head
        scale_patch_num = self.scale or 1. / sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num,
                                        keys_patch_num)  # batch*ch, nheads, p_size, p_size
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1))  # B*D_model H S S

        # Upsampling
        series_patch_size = repeat(series_patch_size, 'b l m n -> b l (m repeat_m) (n repeat_n)',
                                   repeat_m=self.patch_size[patch_index], repeat_n=self.patch_size[patch_index])
        series_patch_num = series_patch_num.repeat(1, 1, self.window_size // self.patch_size[patch_index],
                                                   self.window_size // self.patch_size[patch_index])
        series_patch_size = reduce(series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
        series_patch_num = reduce(series_patch_num, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)

        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return (None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(B, L, H, -1)
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(B, L, H, -1)

        # patch_num
        B, L, M = x_patch_num.shape
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(B, L, H, -1)
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(B, L, H, -1)

        # x_ori
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)

        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            values, patch_index,
            attn_mask
        )

        return series, prior


class ConvTransformerForCon(nn.Module):
    def __init__(self, input_channels, intermediate_channels, output_channels):
        super(ConvTransformerForCon, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(intermediate_channels, output_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(output_channels, 38, kernel_size=1)  # 将通道数直接转换为 51
        self.pool = nn.AdaptiveAvgPool2d((100, 256))  # 自适应池化到 (100, 256)

    def forward(self, x):
        print("x.shape in ConvTransformerForCon_1:",x.shape)
        # 使用卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # 将通道数直接转换为 51

        # 池化到 (100, 256)
        x = self.pool(x)

        # 调整形状到 (256, 100, 51)
        x = x.permute(0, 2, 3, 1).reshape(256, 100, -1)

        return x

class DimReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimReducer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x 的形状: [256, 100, 459]
        print("x.shape in DimReducer_2:",x.shape)
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(-1, feature_dim)  # 形状: [25600, 459]
        x = self.fc(x)  # 形状: [25600, 51]
        x = x.view(batch_size, seq_len, -1)  # 形状: [256, 100, 51]
        a,b,c = x.shape
        # print("modules---1",x.shape)
        # if a != 256:
        #     linear_layer = nn.Linear(a, 256)
        #     reshaped_tensor = x.view(-1, a)  # 结果形状为 [100*51, 226]
        #     transformed_tensor = linear_layer(reshaped_tensor)  # 结果形状为 [100*51, 256]
        #     x = transformed_tensor.view(256, 100, 51)

        return x
