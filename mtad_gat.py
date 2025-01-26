import torch
import torch.nn as nn
from einops import rearrange
from tkinter import _flatten


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
        n_features,  
        window_size,  
        out_dim,   
        kernel_size=7, 
        feat_gat_embed_dim=None,   
        time_gat_embed_dim=None,  
        use_gatv2=True,    
        gru_n_layers=1,   
        gru_hid_dim=150,    
        forecast_n_layers=1,   
        forecast_hid_dim=150,  
        recon_n_layers=1,   
        recon_hid_dim=150,  
        dropout=0.2,
        alpha=0.2, 

        subgraph_size=20, 
        node_dim=40,
        device='cuda',
        propalpha=0.05,
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout) # 通过RNN实现重构

        self.single_step=True
        self.layer_num=3   
        self.gcn_depth = 2  
        self.conv_channels=100   
        self.gnn_channels = [8, 16, 32, 64]  
        self.scale_channels=[8, 16, 32, 64]  
        self.gc = graph_constructor(n_features, subgraph_size, node_dim, self.layer_num, device)  
        self.idx = torch.arange(n_features).to(device)  
        self.scale_idx = torch.arange(n_features).to(device) 
        self.seq_length = 24*7 
        self.gconv1 = nn.ModuleList() 
        self.gconv2 = nn.ModuleList()  
        self.scale_convs = nn.ModuleList()  
        if self.single_step:    
            self.kernel_set = [7, 6, 3, 2]  
        else:
            self.kernel_set = [3, 2, 2]
        length_set = []
        length_set.append(self.seq_length - self.kernel_set[0] + 1) 
        for i in range(1, self.layer_num):
            length_set.append( int( (length_set[i-1]-self.kernel_set[i])/2 ) )
        for i in range(self.layer_num):       
            self.gconv1.append(mixprop(self.conv_channels, self.gnn_channels[i], self.gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(self.conv_channels, self.gnn_channels[i], self.gcn_depth, dropout, propalpha))
            self.scale_convs.append(nn.Conv2d(in_channels=self.conv_channels, # 尺度卷积层
                                                    out_channels=self.scale_channels[i],
                                                    kernel_size=(1, length_set[i])))
            self.linear_model = Conv1DReshape()

        d_model = 256
        win_size = window_size
        patch_size = [4, 5, 10]
        self.patch_size = patch_size
        self.win_size = win_size
        enc_in = n_features  
        c_out = n_features   
        channel = n_features    
        output_attention = True
        e_layers = 3
        n_heads = 1
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))
        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
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

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        k1,k2,k3 = x.shape
        t = x
        series_patch_mean = []
        prior_patch_mean = []
        x_ori = self.embedding_window_size(x)
        # print(x.shape)
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size
        
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)
        
        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))

        series_patch_mean_tensor = torch.stack(series_patch_mean)
        prior_patch_mean_tensor = torch.stack(prior_patch_mean) 
        x = torch.cat((series_patch_mean_tensor, prior_patch_mean_tensor), dim=3) 
        x = torch.squeeze(x)
        b1, n, h, w = x.shape
        merged_tensor = x.permute(1, 3, 0, 2) 
        c1, c2, c3, c4 = merged_tensor.shape 
        new_k2_d = c3 * c4 
        x = merged_tensor.reshape(c1, k2, new_k2_d)
        linear = nn.Linear(new_k2_d, k3).cuda()
        x = linear(x) 
        x = x+t
        self.scale_set = [1, 0.8, 0.6, 0.5]
        adj_matrix = self.gc(self.idx, self.scale_idx,self.scale_set)
        scale = []
        outputs = []
        for i in range(self.layer_num):
          scale.append(x)
        for i in range(self.layer_num): 
            output = self.gconv1[i](scale[i], adj_matrix[i]) + self.gconv2[i](scale[i], adj_matrix[i].transpose(1, 0))
            outputs.append(output)
        outputs = torch.cat(outputs,dim=1)
        outputs = torch.squeeze(outputs)
        outputs = self.linear_model(outputs)
        x = outputs 
        
        x = x+t
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1) 
        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end) 

        return predictions, recons
