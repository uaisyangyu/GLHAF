import torch
import torch.nn as nn
import torch.nn.functional as F



class GFC(nn.Module):
    def __init__(self, d, temp=1.0):
        super().__init__()
        self.temp = temp
        self.norm_g = nn.LayerNorm(d)
        self.norm_l = nn.LayerNorm(d)
        self.fuse   = nn.Linear(2*d, d)          # ③ 可选：再精炼

    def cos_gate(self, l, g):
        l = F.normalize(l, dim=-1)   # (L,B,d)
        g = F.normalize(g, dim=-1)
        cos = (l * g).sum(-1, keepdim=True)  # (L,B,1)
        return torch.sigmoid((1 - cos) * self.temp)

    def forward(self, G_plus, l_t, l_a, l_v):
        # ① 超全局 G⁺
        G_plus = self.norm_g(G_plus)

        # ②③ 逐模态差异-互补
        def refine(l):
            l = self.norm_l(l)
            D = self.cos_gate(l, G_plus)
            return D * l                                  # 只留“独有”

        l_plus_t = refine(l_t)
        l_plus_a = refine(l_a)
        l_plus_v = refine(l_v)

        # ④ 拼回
        out = torch.cat([l_plus_t, l_plus_a, l_plus_v], 0)  # 3L,B,3)
        return out


# class GlobalFusionDiffComplement(nn.Module):
#     def __init__(self, d, temp=4.0):
#         super().__init__()
#         self.temp = temp
#         self.to_d   = nn.Linear(3*d, d)          # ① 超全局降维
#         self.norm_g = nn.LayerNorm(d)
#         self.norm_l = nn.LayerNorm(d)
#         self.fuse   = nn.Linear(2*d, d)          # ③ 可选：再精炼
#
#     def cos_gate(self, l, g):
#         l = F.normalize(l, dim=-1)   # (L,B,d)
#         g = F.normalize(g, dim=-1)
#         cos = (l * g).sum(-1, keepdim=True)  # (L,B,1)
#         return torch.sigmoid((1 - cos) * self.temp)
#
#     def forward(self, G_t, G_a, G_v, l_t, l_a, l_v,return_gate=False):
#         # ① 超全局 G⁺
#         G_cat  = torch.stack([G_t, G_a, G_v], 0)          # (3,L,B,d)
#         G_plus = G_cat.permute(1,2,0,3).reshape(G_t.size(0), -1, 3*G_t.size(-1))
#         G_plus = self.to_d(G_plus)                        # (L,B,d)
#         G_plus = self.norm_g(G_plus)
#
#         # ②③ 逐模态差异-互补
#         def refine(l):
#             l = self.norm_l(l)
#             D = self.cos_gate(l, G_plus)
#             return D * l,D.squeeze(-1)                                # 只留“独有”
#
#         l_plus_t,D_t = refine(l_t)
#         l_plus_a,D_a = refine(l_a)
#         l_plus_v,D_v= refine(l_v)
#
#         # ④ 拼回
#         out = torch.cat([l_plus_t, l_plus_a, l_plus_v], 0)  # (3L,B,d)
#         if return_gate:
#             # 堆成 (3,L,B) 方便后续可视化
#             gate = torch.stack([D_t, D_a, D_v], 0)  # (3,L,B)
#             return out, gate
#         return out
#
# G_t, G_a, G_v = torch.rand(12, 12, 36), torch.rand(12, 12, 36), torch.rand(12, 12, 36)
# l_t, l_a, l_v = torch.rand(12, 12, 36), torch.rand(12, 12, 36), torch.rand(12, 12, 36)
#
# glc = GlobalFusionDiffComplement(d=36)
# refined_local, gate = glc(G_t, G_a, G_v, l_t, l_a, l_v, return_gate=True)
# # refined_local: (3L,B,d)  gate: (3,L,B)  -> 可视化/分析
# print(gate.size())
