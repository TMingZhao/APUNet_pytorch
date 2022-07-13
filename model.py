import torch.nn as nn
import torch
from utils.loss import get_repulsion_loss, get_emd_loss
from utils.transformer_o import TransformerDecoder, TransformerEncoder, TransformerDecoderLayer, TransformerEncoderLayer, _get_clones
from utils.point_util import square_distance
import config
import torch.nn.functional as F


class feature_fuse_1(nn.Module):
    def __init__(self, local_dim=128, global_dim=256, fuse_dim=128):
        super(feature_fuse_1, self).__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.fuse_dim = fuse_dim
        self.fuse = nn.Sequential(
            nn.Linear(self.local_dim+self.global_dim, self.fuse_dim),
            nn.ReLU(),
        )
        
    def forward(self, local_feature, global_feature):
        # local_feature: [Bm, n, c]    [B, n, c]
        # global_feature: [B, m, d]    [1, B, D]
        B, m, _ = global_feature.shape  # [1, B, D]
        _, n, _ = local_feature.shape   # [B, n, c]

        local_feature = local_feature.reshape(B, m, n, -1)  # [B, m, n, C]  # [1, B, n, -1]
        global_feature = global_feature.unsqueeze(2).repeat(1, 1, n, 1)     # [B, m, D] --> [B, m, n, D]    # [1, B, 1, D] -> [1, B, n, D]
        fuse_feature = torch.cat((local_feature, global_feature), dim=3)        # [B, m, n, C+D]    # [1, B, n, C+D]
        fuse_feature = self.fuse(fuse_feature)      # [B, m, n, e]      # [1, B, n, e]
        fuse_feature = fuse_feature.reshape(B*m, n, -1) # [Bm, n, e]    # [B, n, e]

        return fuse_feature

class MultiFeatureFusion(nn.Module):
    def __init__(self, local_dim=128, global_dim=256, fuse_dim=128, fusion_layer_idxs=[1, 3, 7]):
        super(MultiFeatureFusion, self).__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.fuse_dim = fuse_dim
        self.fusion_layer_idxs = fusion_layer_idxs
        fuse_layer = feature_fuse_1(self.local_dim, self.global_dim, self.fuse_dim)
        self.fuse_layers = _get_clones(fuse_layer, len(self.fusion_layer_idxs))
        
    def forward(self, local_features, global_features):
        # local_feature: [s, Bm, n, c]
        # global_feature: [s, B, m, d].
        intermediate = []
        for (idx, fuse_layer) in zip(self.fusion_layer_idxs, self.fuse_layers):
            fuse_feature = fuse_layer(local_features[idx], global_features[idx])
            intermediate.append(fuse_feature)

        return intermediate
    
class TPU(nn.Module):
    def __init__(self, input_point_num=32, fix_bone=False, up_ratio=4, visual_dir=''):
        super(TPU, self).__init__()
        self.local_dim = config.local_dim
        self.global_dim = config.global_dim
        self.fuse_dim = config.fuse_dim 
        ### local feature extract
        self.feature_encoding = nn.Linear(3, self.local_dim)
        self.positional_encoding = nn.Linear(3, self.local_dim)
        self.local_layer_norm = nn.LayerNorm(self.local_dim)
        local_encoder_layer = TransformerEncoderLayer(self.local_dim, config.nhead, self.local_dim*4,
                                                config.dropout, activation="relu", normalize_before=True)
        self.local_layers = TransformerEncoder(local_encoder_layer, config.num_encoder, self.local_layer_norm, return_intermediate=True)
        
        ### global feature extract
        self.trans_dimen = nn.Conv1d(self.local_dim, self.global_dim, 1)    # [B,N,16]
        self.maxpool = nn.AdaptiveMaxPool1d(1) # 平均池化操作
        self.global_pos = nn.Conv1d(4, self.global_dim, 1)

        global_encoder_layer = TransformerEncoderLayer(self.global_dim, config.nhead, dim_feedforward=2048 if self.global_dim>512 else self.global_dim*4,
                                                            dropout=config.dropout, activation="relu", normalize_before=True)
        self.global_layer_norm = nn.LayerNorm(self.global_dim)   
        self.global_layers = TransformerEncoder(global_encoder_layer, config.num_encoder, self.global_layer_norm, return_intermediate=True)
        ### feature fuse
        fuse_idxes = [1, 3, 7]
        self.fuse_layers = MultiFeatureFusion(self.local_dim, self.global_dim, self.fuse_dim, fuse_idxes)
        self.fuse_weights = nn.Parameter(torch.ones(len(fuse_idxes)))
        ### feature prediction
        pred_dim = self.fuse_dim
        self.query_embed = nn.Embedding(
            num_embeddings=config.mini_point * config.up_ratio,
            embedding_dim=pred_dim
        )

        self.tgt = nn.Parameter(torch.rand((config.mini_point * config.up_ratio, pred_dim)))
        decoder_layer = TransformerDecoderLayer(pred_dim, config.nhead, pred_dim*4,
                                                config.dropout, activation="relu", normalize_before=True)
        decoder_norm = nn.LayerNorm(pred_dim)
        self.decoder = TransformerDecoder(decoder_layer, config.num_decoder, decoder_norm, return_intermediate=True)

        ### coordinate regression
        self.coord_layer = nn.Sequential(
            nn.Conv1d(pred_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

        self.init_weight()

    def init_weight(self):
        # 参数初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)
        
    def get_dis_correlation_matrix(self, xyz):
        # xyz: [B, N, 3]

        sqrdists = square_distance(xyz, xyz)      # [B, N, N]
        # Remove negative values due to calculation errors
        sqrdists = torch.clamp(sqrdists,min=0.0)
        dist = torch.sqrt(sqrdists)      # [B, N, N]
        
        # 采用分段函数，when d<dist, corr = 1；when d>dist，mask = (1-min_)*(dist-2)/(thres-2) + min_
        thres, min_ = config.cor_thres, config.cor_min
        corr = (1-min_)*(dist-2)/(thres-2) + min_
        corr = torch.clamp(corr, min=1e-9, max=1.0)
        corr = torch.log(corr)

        return corr

    def forward(self, xyz, gt=None, patch_pos=None, img=None, gt_box=None):
        # xyz: [B, 3, N]
        # gt: [B, 3, rN]
        # patch_pos: [B, 4, 1]
        xyz = xyz.permute(0,2,1)
        B, N, _ = xyz.shape

        # input 局部特征提取的输入为归一化的点云面片
        local_feature = self.feature_encoding(xyz)   # [B, n, c]
        pos = self.positional_encoding(xyz)   # [B, n, c]
        local_corr = self.get_dis_correlation_matrix(xyz)     # [B, n, n]
        local_features = self.local_layers(local_feature, corr=local_corr, pos=pos) # [s, B, n, c]

        global_feature = local_feature
        global_feature = self.trans_dimen(global_feature.permute(0, 2, 1))   # [B, D, n]
        global_feature = self.maxpool(global_feature)   # [B, D, 1]
        global_feature = global_feature.reshape(1, B, global_feature.shape[1])    # [1, B, D]
        globale_corr = self.get_dis_correlation_matrix(patch_pos.permute(2, 0, 1)[:,:,0:3])  # [1, B, B]
        global_pos = self.global_pos(patch_pos).permute(2, 0, 1) # [1, B, D]
        global_features = self.global_layers(global_feature, corr=globale_corr, pos=global_pos) # [s, 1, B, D]

        fuse_features = self.fuse_layers(local_features, global_features) #  [s, B, n, c]
        f_weight = F.softmax(self.fuse_weights, dim=0)    # [s]

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s
        fuse_feature = weighted_avg(fuse_features, f_weight)  # [B, n, e]
        ##  feature predict
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(fuse_feature.shape[0], 1, 1)  # [B, rn, E]
        tgt = self.tgt.unsqueeze(0).repeat(fuse_feature.shape[0], 1, 1)  # [B, rn, E]
        feature = self.decoder(tgt, fuse_feature, memory_mask=None,
                          pos=pos, query_pos=query_embed)[-1]   # [B, rn, E]
        
        ## Coordinate regression
        coord = self.coord_layer(feature.permute(0, 2, 1))   # [B, 3, rn]
        if gt is None:
            return coord
        else:
            loss4 = get_repulsion_loss(coord.permute(0,2,1), h=0.0705)
            loss5 = get_emd_loss(coord.permute(0,2,1).contiguous(), gt.permute(0,2,1).contiguous())
            loss =  loss5 + 0.2 * loss4
            loss_log = {
                'emd_loss': loss5.cpu().detach().item(),
                'rep_loss': loss4.cpu().detach().item(),
            }
            return loss, coord, loss_log
        




