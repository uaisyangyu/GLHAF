import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.wav2vec2.components import SelfAttention

from ...subNets.BertTextEncoder import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
    # from ...subNets.OT import OTLocalFusion
from ...subNets.OTDBFusion import OTDBFusion
from ...subNets.Bottleneck import BottleneckFusion
from .fusion import GFC
class G2L(nn.Module):
        def __init__(self, args):
            super(G2L, self).__init__()
            if args.use_bert:
                self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                                  pretrained=args.pretrained)
            self.use_bert = args.use_bert
            dst_feature_dims, nheads = args.dst_feature_dim_nheads  # [50,10]
            if args.dataset_name == 'mosi':
                if args.need_data_aligned:
                    self.len_l, self.len_v, self.len_a = 50, 50, 50
                else:
                    self.len_l, self.len_v, self.len_a = 50, 500, 375
            if args.dataset_name == 'mosei':
                if args.need_data_aligned:
                    self.len_l, self.len_v, self.len_a = 50, 50, 50
                else:
                    self.len_l, self.len_v, self.len_a = 50, 500, 500
            self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims  # [768, 5, 20]
            self.d_l = self.d_a = self.d_v = dst_feature_dims  # [150]
            self.num_heads = nheads
            self.layers = args.nlevels  # 2
            self.attn_dropout = args.attn_dropout
            self.attn_dropout_a = args.attn_dropout_a
            self.attn_dropout_v = args.attn_dropout_v
            self.relu_dropout = args.relu_dropout
            self.embed_dropout = args.embed_dropout
            self.res_dropout = args.res_dropout
            self.output_dropout = args.output_dropout
            self.text_dropout = args.text_dropout
            self.attn_mask = args.attn_mask
            combined_dim_low = self.d_a
            combined_dim_high = self.d_a
            combined_dim = (self.d_l + self.d_a + self.d_v)
            output_dim = 1
            num_bottleneck = 10
            # 1. Temporal convolutional layers for initial feature#将特征统一到同一维度
            self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0,
                                    bias=False)  # 卷积核为5
            self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0,
                                    bias=False)
            self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0,
                                    bias=False)

            # 2. Global feature encoder
            self.encoder_g = self.get_network(self_type='l', layers=self.layers)
            # self.g_proj_l = nn.AdaptiveMaxPool1d(output_size=self.len_l - args.conv1d_kernel_size_l + 1)
            # self.g_proj_v = nn.AdaptiveMaxPool1d(output_size=self.len_v - args.conv1d_kernel_size_v + 1)
            # self.g_proj_a = nn.AdaptiveMaxPool1d(output_size=self.len_a - args.conv1d_kernel_size_a + 1)
            # 3. Local feature encoder
            self.encoder_s_l = self.get_network(self_type='l', layers=self.layers)
            self.encoder_s_v = self.get_network(self_type='v', layers=self.layers)
            self.encoder_s_a = self.get_network(self_type='a', layers=self.layers)

            # for align g_l, g_v, g_a(对齐全局特征）
            self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                       combined_dim_low)  # （50×46，50）
            self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                       combined_dim_low)
            self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                       combined_dim_low)

            self.BFusion = BottleneckFusion(dim=dst_feature_dims, num_bottleneck=20, num_layers=3)
            # for align s_l, s_v, s_a(对齐局部特征）
            # self.align_local_tool=LocalOTAlign(eps=0.5,hard=True)

            # 4 Multimodal Crossmodal Attentions
            self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
            self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
            self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
            self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
            self.trans_g_with_l = self.get_network(self_type='gl', layers=self.layers)
            self.trans_g_with_v = self.get_network(self_type='gv', layers=self.layers)
            self.trans_g_with_a = self.get_network(self_type='ga', layers=self.layers)

            self.self_attentions_g = self.get_network(self_type='l')
            self.self_attentions_l = self.get_network(self_type='l')
            self.self_attentions_v = self.get_network(self_type='v')
            self.self_attentions_a = self.get_network(self_type='a')

            self.proj1_l = nn.Linear(self.d_l, self.d_l)
            self.proj2_l = nn.Linear(self.d_l, self.d_l)
            self.out_layer_l = nn.Linear(self.d_l, output_dim)
            self.proj1_v = nn.Linear(self.d_v, self.d_v)
            self.proj2_v = nn.Linear(self.d_v, self.d_v)
            self.out_layer_v = nn.Linear(self.d_v, output_dim)
            self.proj1_a = nn.Linear(self.d_a, self.d_a)
            self.proj2_a = nn.Linear(self.d_a, self.d_a)
            self.out_layer_a = nn.Linear(self.d_a, output_dim)
            # 全局降维
            self.to_d = nn.Linear(self.d_l * 3, self.d_l)
            self.GFC = GFC(d=self.d_l)
            # self.OTFusion = OTLocalFusion(d=self.d_l)
            self.OTDBFusion = OTDBFusion(d=self.d_l)
            # 5. fc layers for global features
            self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                         combined_dim_low)
            self.proj2_l_low = nn.Linear(combined_dim_low,
                                         combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
            self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                             output_dim)
            self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                         combined_dim_low)
            self.proj2_v_low = nn.Linear(combined_dim_low,
                                         combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
            self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                             output_dim)
            self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                         combined_dim_low)
            self.proj2_a_low = nn.Linear(combined_dim_low,
                                         combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
            self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                             output_dim)

            # 6. fc layers for local features
            self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
            self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
            self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
            self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

            # 7. project for fusion
            self.projector_l = nn.Linear(self.d_l, self.d_l)
            self.projector_v = nn.Linear(self.d_v, self.d_v)
            self.projector_a = nn.Linear(self.d_a, self.d_a)
            self.projector_g = nn.Linear(self.d_l, self.d_l)

            self.proj1_g = nn.Linear(self.d_l, self.d_l)
            self.proj2_g = nn.Linear(self.d_l, self.d_l)
            self.out_layer_g = nn.Linear(self.d_l, output_dim)

            # 8. final project
            self.f_proj1 = nn.Linear(self.d_l, self.d_l)
            self.f_proj2 = nn.Linear(self.d_l, self.d_l)
            self.out_layer_two = nn.Linear(self.d_l, output_dim)
            self.xz = nn.Linear(self.d_l, self.d_l)
            # 原始
            self.proj1 = nn.Linear(combined_dim, combined_dim)
            self.proj2 = nn.Linear(combined_dim, combined_dim)
            self.out_layer = nn.Linear(combined_dim, output_dim)

        def get_network(self, self_type='l', layers=-1):
            if self_type in ['l', 'al', 'vl', 'gl']:
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type in ['a', 'la', 'va', 'ga']:
                embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
            elif self_type in ['v', 'lv', 'av', 'gv']:
                embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
            elif self_type == 'l_mem':
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type == 'a_mem':
                embed_dim, attn_dropout = self.d_a, self.attn_dropout
            elif self_type == 'v_mem':
                embed_dim, attn_dropout = self.d_v, self.attn_dropout
            else:
                raise ValueError("Unknown network type")

            return TransformerEncoder(embed_dim=embed_dim,
                                      num_heads=self.num_heads,
                                      layers=max(self.layers, layers),
                                      attn_dropout=attn_dropout,
                                      relu_dropout=self.relu_dropout,
                                      res_dropout=self.res_dropout,
                                      embed_dropout=self.embed_dropout,
                                      attn_mask=self.attn_mask)

        def forward(self, text, audio, video):
            # extraction
            if self.use_bert:
                text = self.text_model(text)
            x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout,
                            training=self.training)  # [batch_size, seq_len, text_dim]->[batch_size, text_dim, seq_len]
            x_a = audio.transpose(1, 2)
            x_v = video.transpose(1, 2)

            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(
                x_l)  # [batch_size, 150, input_len-5+1]
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)  # [5,50,5]
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)  # [20,50,5]

            # g_x_l = self.g_proj_l(proj_x_l)
            # g_x_a = self.g_proj_a(proj_x_a)
            # g_x_v = self.g_proj_v(proj_x_v)

            # 在trasnformer模块需要输入的格式是[lenth,batch_siaze,dim]的形状
            proj_x_l = proj_x_l.permute(2, 0, 1)  # [outlen=length-5+1,batch_size, dim[50]]
            proj_x_v = proj_x_v.permute(2, 0, 1)
            proj_x_a = proj_x_a.permute(2, 0, 1)

            # local feature
            s_l = self.encoder_s_l(proj_x_l)  # [lenth,batch_size,dim=50)
            s_v = self.encoder_s_v(proj_x_v)
            s_a = self.encoder_s_a(proj_x_a)

            local_l = s_l.permute(1, 0, 2)  # [batch,length,dim]
            local_v = s_v.permute(1, 0, 2)
            local_a = s_a.permute(1, 0, 2)

            local_align_l = local_l.permute(1, 0, 2)  # [length,batch,dim]
            local_align_v = local_v.permute(1, 0, 2)
            local_align_a = local_a.permute(1, 0, 2)  # [46, 28, 140]

            # gobal feature
            c_l = self.encoder_g(proj_x_l)  # [lenth,batch,dim]
            c_v = self.encoder_g(proj_x_v)
            c_a = self.encoder_g(proj_x_a)
            # c_l = self.encoder_g(g_x_l.permute(2, 0, 1))  # [lenth,batch,dim]
            # c_v = self.encoder_g(g_x_v.permute(2, 0, 1))
            # c_a = self.encoder_g(g_x_a.permute(2, 0, 1))

            c_l = c_l.permute(1, 2, 0)  # [batch,dim,length]
            c_v = c_v.permute(1, 2, 0)
            c_a = c_a.permute(1, 2, 0)
            c_list = [c_l, c_v, c_a]

            # 用于全局模态对齐，CMD损失,展开计算dim*length
            # g_align_l=c_l.contiguous().view(x_l.size(0),-1)#[batch_size,dim[50]*length]
            # g_align_v=c_v.contiguous().view(x_v.size(0),-1)
            # g_align_a=c_a.contiguous().view(x_a.size(0),-1)
            # 用于三重损失对齐
            # self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low) (combined_dim_low[50] * length, combined_dim_low)
            c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0),
                                                           -1))  # [batch_size,dim[50]*length]经过线性变换[batch_size,combined_dim_low[50]]
            c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
            c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

            c_l = c_l.permute(2, 0, 1)  # [length,batch,dim]
            c_v = c_v.permute(2, 0, 1)
            c_a = c_a.permute(2, 0, 1)

            # 信息瓶颈融合
            # g_fusion = self.BFusion(c_l, c_v, c_a)
            # g_fusion=torch.cat([c_l, c_v, c_a], dim=0)#按长度凭借
            g_fusion = torch.cat([c_l, c_v, c_a], dim=2)
            g_fusion = self.to_d(g_fusion)
            # print(g_fusion.size())

            g_att = self.self_attentions_g(g_fusion)
            if type(g_att) == tuple:
                g_att = g_att[0]
            g_att = g_att[-1]  # [batch_size,dim]

            g_proj = self.proj2_g(
                F.dropout(F.relu(self.proj1_g(g_att), inplace=True), p=self.output_dropout,
                          training=self.training))
            g_proj += g_att
            logits_g = self.out_layer_g(g_proj)  # [batch,outdim=1]

            # 局部预测能力
            l_att = self.self_attentions_l(local_align_l)
            if type(l_att) == tuple:
                l_att = l_att[0]
            l_att = l_att[-1]

            l_proj = self.proj2_l(F.dropout(F.relu(self.proj1_l(l_att), inplace=True), p=self.output_dropout,
                                            training=self.training))
            l_proj += l_att
            logits_l = self.out_layer_l(l_proj)
            # 视觉
            v_att = self.self_attentions_v(local_align_v)
            if type(v_att) == tuple:
                v_att = v_att[0]
            v_att = v_att[-1]

            v_proj = self.proj2_v(F.dropout(F.relu(self.proj1_v(v_att), inplace=True), p=self.output_dropout,
                                            training=self.training))
            v_proj += v_att
            logits_v = self.out_layer_v(v_proj)
            # 音频
            a_att = self.self_attentions_a(local_align_a)
            if type(a_att) == tuple:
                a_att = a_att[0]
            a_att = a_att[-1]

            a_proj = self.proj2_a(F.dropout(F.relu(self.proj1_a(a_att), inplace=True), p=self.output_dropout,
                                            training=self.training))
            a_proj += a_att
            logits_a = self.out_layer_a(a_proj)

            # 局部和全局融合
            # # g-->l
            h_g = g_fusion
            L_dict = {'v': local_align_v, 't': local_align_l, 'a': local_align_a}
            #
            h_orgin = torch.cat([L_dict['v'], L_dict['t'], L_dict['a']], dim=0)
            result = self.OTDBFusion(h_g, L_dict)

            C_v, C_a, C_t = result['C_dict']['v'], result['C_dict']['a'], result['C_dict']['t']
            h_v, h_a, h_t = result['S_dict']['v'], result['S_dict']['a'], result['S_dict']['t']

            # hf_local_s = torch.cat([result['S_dict']['v'], result['S_dict']['t'], result['S_dict']['a']], dim=0)

            # 只有差异
            # hf_local=self.OTFusion(L_dict,h_g)
            # h_gf = torch.cat([h_g, hf_local], dim=0)

            #
            # hf_local=self.GFC(h_g,local_align_l,local_align_v,local_align_a)
            #
            h_g_t = self.trans_g_with_l(h_g, h_t, h_t)
            h_g_v = self.trans_g_with_v(h_g, h_v, h_v)
            h_g_a = self.trans_g_with_v(h_g, h_a, h_a)
            C_hv = torch.bmm(C_v, h_g_v.transpose(0, 1)).transpose(0, 1)
            C_ha = torch.bmm(C_a, h_g_a.transpose(0, 1)).transpose(0, 1)
            C_ht = torch.bmm(C_t, h_g_t.transpose(0, 1)).transpose(0, 1)
            hf_local_s = torch.cat([C_ht, C_hv, C_ha], dim=0)

            # h_gf=self.trans_g_with_l(h_g,h_orgin,h_orgin)
            # h_gf=torch.cat([h_gf,hf_local_s],dim=0)
            h_gf = self.trans_l_mem(hf_local_s)
            if type(h_gf) == tuple:
                h_gf = h_gf[0]
            last_gf = h_gf[-1]

            hs_proj_fl_high = self.proj2_l_high(
                F.dropout(F.relu(self.proj1_l_high(last_gf), inplace=True), p=self.output_dropout,
                          training=self.training))
            hs_proj_fl_high += last_gf
            logits_fl_high = self.out_layer_l_high(hs_proj_fl_high)

            last_hs = torch.sigmoid(self.projector_l(hs_proj_fl_high))
            # prediction
            last_hs_proj = self.f_proj2(
                F.dropout(F.relu(self.f_proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.xz(last_hs_proj)
            last_hs_proj += last_hs

            output = self.out_layer_two(last_hs_proj)

            res = {
                'origin_l': proj_x_l,
                'origin_v': proj_x_v,
                'origin_a': proj_x_a,
                'c_l': c_l,
                'c_v': c_v,
                'c_a': c_a,
                's_l': s_l,
                's_v': s_v,
                's_a': s_a,
                'g_proj': g_proj,
                'l_proj': l_proj,
                'v_proj': v_proj,
                'a_proj': a_proj,
                'g_fusion': g_fusion,
                'local_l': local_l,
                'local_a': local_a,
                'local_v': local_v,
                'local_align_l': local_align_l,
                'local_align_a': local_align_a,
                'local_align_v': local_align_v,
                'c_l_sim': c_l_sim,
                'c_v_sim': c_v_sim,
                'c_a_sim': c_a_sim,
                'logits_l': logits_l,
                'logits_v': logits_v,
                'logits_a': logits_a,
                'logits_g': logits_g,
                'logits_fl_high': logits_fl_high,
                'last_hs_proj': last_hs_proj,
                'output_logit': output
            }
            return res
