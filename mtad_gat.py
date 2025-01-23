import torch
# torch.cuda.empty_cache()
import torch.nn as nn
# next from contrastive
from einops import rearrange
from tkinter import _flatten
# above come from contrastive

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
    graph_constructor,
    mixprop,
    LinearForReshape,
    Conv1DReshape,
    PositionalEmbedding,
    TokenEmbedding,
    DataEmbedding,
    Encoder,
    DAC_structure,
    AttentionLayer,
    ConvTransformerForCon,
    DimReducer,
)


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,     # 特征数量
        window_size,    # 窗口大小
        out_dim,    # 特征输出维度
        kernel_size=7,  # 卷积核大小
        feat_gat_embed_dim=None,    # 特征层的线性变换输出维度
        time_gat_embed_dim=None,    # 时间层的线性变换输出维度
        use_gatv2=True,     # 修改后的注意力机制
        gru_n_layers=1,     # GRU层数
        gru_hid_dim=150,    # gru隐藏层维度
        forecast_n_layers=1,     # 全连接预测层数
        forecast_hid_dim=150,   # 全连接预测维度
        recon_n_layers=1,    # gru重构层数
        recon_hid_dim=150,  # gru重构维度
        dropout=0.2,
        alpha=0.2,  # leaky rely激活函数的斜率参数

        subgraph_size=20,   # 子图尺寸:k,subgraph_size = params['subgraph_size']
        node_dim=40,
        device='cuda',
        # device='cpu',
        propalpha=0.05,
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout) # 通过RNN实现重构

        # 初始化邻接矩阵
        # self.seq_length = seq_length  # 输入序列的长度
        self.single_step=True
        self.layer_num=3    # 层数
        self.gcn_depth = 2  # 图卷积深度
        # self.conv_channels=[8, 16, 32, 64]   # conv_channels = params['conv_channels']，以下三个，手动调参
        self.conv_channels=100   # conv_channels = params['conv_channels']，以下三个，手动调参
        self.gnn_channels = [8, 16, 32, 64]  # gnn_channels = conv_channels,"conv_channels":{"_type":"choice","_value":[8, 16, 32, 64]}
        # self.scale_channels=16  # scale_channels = conv_channels
        self.scale_channels=[8, 16, 32, 64]  # scale_channels = conv_channels
        self.gc = graph_constructor(n_features, subgraph_size, node_dim, self.layer_num, device)  # 多层次林截图构造器，形成邻接矩阵
        self.idx = torch.arange(n_features).to(device)  # 索引，数量等于特征数量
        self.scale_idx = torch.arange(n_features).to(device)  # 尺度索引张量,数量等于节点数量
        self.seq_length = 24*7 # 输入序列的长度
        # self.seq_length = 12  # 输入序列的长度
        self.gconv1 = nn.ModuleList()  # 用于保存第一个图卷积层
        self.gconv2 = nn.ModuleList()  # 用于保存第二个图卷积层
        self.scale_convs = nn.ModuleList()  # 用于保存尺度卷积层
        if self.single_step:    # 根据single_step设定卷积核大小
            self.kernel_set = [7, 6, 3, 2]  # kernel_size=args.kernel_size，还是可以手动调节，选择最合适的，固定尺度
        else:
            self.kernel_set = [3, 2, 2]
        length_set = []
        length_set.append(self.seq_length - self.kernel_set[0] + 1)  # 第一个卷积核的尺寸
        for i in range(1, self.layer_num): # 模块a：从小尺度卷积到大尺度 根据卷积核大小和层数计算每一层的卷积长度
            length_set.append( int( (length_set[i-1]-self.kernel_set[i])/2 ) ) # 卷积核慢慢变成大尺度
        for i in range(self.layer_num): # 模块c：融合特定尺度表示和邻接矩阵并获取特定的尺度表示
            """
            RNN based model
            """
            # self.agcrn.append(AGCRN(num_nodes=self.num_nodes, input_dim=conv_channels, hidden_dim=scale_channels, num_layers=1) )
            self.gconv1.append(mixprop(self.conv_channels, self.gnn_channels[i], self.gcn_depth, dropout, propalpha)) # 获取特定尺度表示，融合邻接矩阵和尺度表示
            self.gconv2.append(mixprop(self.conv_channels, self.gnn_channels[i], self.gcn_depth, dropout, propalpha))
            self.scale_convs.append(nn.Conv2d(in_channels=self.conv_channels, # 尺度卷积层
                                                    out_channels=self.scale_channels[i],
                                                    kernel_size=(1, length_set[i])))
            # self.linear_model = LinearForReshape()
            self.linear_model = Conv1DReshape()

            # ----------初始化邻接矩阵-----------#

        # ----------- next:contrastive study -----------#
        # d_model = 64
        d_model = 256
        win_size = window_size
        patch_size = [4, 5, 10]
        # patch_size = [3, 5, 7]
        self.patch_size = patch_size
        self.win_size = win_size
        enc_in = n_features     # 暂定
        c_out = n_features     # 暂定
        channel = n_features    # 暂定
        output_attention = True
        e_layers = 3
        n_heads = 1
        # init contrastive
        # Patching List
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout,
                                  output_attention=output_attention),
                    d_model, patch_size, channel, n_heads, win_size) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        # end of contrastive



    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        k1,k2,k3 = x.shape
        t = x

        # 多尺度操作，对比学习，有两个一样的x作为输入
        # Mutil-scale Patching Operation
        # ------begin of CS------
        # series_patch_mean = []
        # prior_patch_mean = []
        # x_ori = self.embedding_window_size(x)
        # # print(x.shape)
        # for patch_index, patchsize in enumerate(self.patch_size):
        #     x_patch_size, x_patch_num = x, x
        #     x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size
        #     x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size
        #
        #     x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)
        #     x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
        #     x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)
        #     x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
        #     # 以上有可能是多尺度卷积，我如果没猜错的话。x_patch_size，x_patch_num，二者应该是相等的。
        #     series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
        #     series_patch_mean.append(series), prior_patch_mean.append(prior)
        #
        # series_patch_mean = list(_flatten(series_patch_mean))
        # prior_patch_mean = list(_flatten(prior_patch_mean))
        # # print("len(series_patch_mean): ", len(series_patch_mean))
        # # print("len(prior_patch_mean:) ", len(prior_patch_mean))
        #
        # # 将展平的列表转换为张量
        # series_patch_mean_tensor = torch.stack(series_patch_mean)  # 假设每个元素是张量
        # prior_patch_mean_tensor = torch.stack(prior_patch_mean)  # 假设每个元素是张量
        # # print(series_patch_mean_tensor.shape)
        # # print(prior_patch_mean_tensor.shape)
        #
        # # 组合张量
        # x = torch.cat((series_patch_mean_tensor, prior_patch_mean_tensor), dim=3)  # 假设沿第1维拼接
        # x = torch.squeeze(x)
        # b1, n, h, w = x.shape
        # merged_tensor = x.permute(1, 3, 0, 2)   # 交换第 1 和第 2 维度
        # c1, c2, c3, c4 = merged_tensor.shape  # 从 merged_tensor 计算 batch size 和其他维度\
        # # print("merged_tensor.shape:",merged_tensor.shape)
        # # merged_tensor.shape: torch.Size([64, 80, 9, 160])
        # # b, n, h, w ：9 64 160 80
        # # k1,k2,k3:256 100 38
        # # b=64
        # # merged_tensor numel: 7372800
        # new_k2_d = c3 * c4 # 80*12800=1024000
        # x = merged_tensor.reshape(c1, k2, new_k2_d)  # 调整为动态的形状
        # # print(f"x shape after reshape   : {x.shape}")
        # # 应用线性变换
        # linear = nn.Linear(new_k2_d, k3).cuda()
        # x = linear(x)  # 这里的输出将自动根据输入的窗口大小和特征维度调整

        #--------end of CS-------

        # --------------from MAGNN work import adjx---------------#
        # print("--------------from MAGNN work import adjx---------------")
        # ---------begin--------
        # x = x+t
        self.scale_set = [1, 0.8, 0.6, 0.5]  # 缩放因子(影响因素，比例)
        # 模块B：输出邻接矩阵，scale_set是缩放因子，用于灵活调整节点的特征，通过forward传递参数
        adj_matrix = self.gc(self.idx, self.scale_idx,self.scale_set)
        # print("000",x.shape)
        # scale:经过多尺度卷积后形成的多尺度特征，层数：3，在这里我可以对input重复添加三次形成
        scale = []
        outputs = []
        for i in range(self.layer_num):
          scale.append(x)
        # print("len(scale),scale[0].shape",len(scale),scale[0].shape)
        for i in range(self.layer_num):     # 融合邻接矩阵与输入的input信息
            output = self.gconv1[i](scale[i], adj_matrix[i]) + self.gconv2[i](scale[i], adj_matrix[i].transpose(1, 0))
            outputs.append(output)
        # print("output is :",outputs)
        # print("output.shape[0]:",outputs[0].shape)
        outputs = torch.cat(outputs,dim=1)
        outputs = torch.squeeze(outputs)
        # print("outputs.shape:",outputs.shape)
        outputs = self.linear_model(outputs)
        # print("outputs.shape:",outputs.shape)
        x = outputs   # 修改+t
        # --------------end of MAGNN work import adjx---------------#

        # print("x.shape (after adj and x):",x.shape) # x.shape (after adj and x): torch.Size([256, 100, 38])
        # error shape:x.shape(after x = self.conv(x)): torch.Size([48, 256, 38])
        x = x+t
        x = self.conv(x)
        # print("x.shape(after x = self.conv(x)):",x.shape)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        # print("x.shape:",x.shape)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp # 重塑形状
        predictions = self.forecasting_model(h_end) # 预测模型，这个要看一下
        recons = self.recon_model(h_end) # 通过rnn来实现重构，重构模型，也要看一下

        return predictions, recons
