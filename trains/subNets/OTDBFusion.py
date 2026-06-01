import torch
import torch.nn as nn
import torch.nn.functional as F
from ot import sinkhorn
from torch.ao.quantization.backend_config.backend_config import CONFIGS_DICT_KEY


class OTDBFusion(nn.Module):
    """
    同时输出：
    - coop_dict:  {'v':[L,B,d], 't':[L,B,d], 'a':[L,B,d]}  协同表示
    - disc_dict:  {'v':[L,B],   't':[L,B],   'a':[L,B]}    差异度
    - G_fused:    [L,B,d]                                 融合后的全局
    - disc_feat:  [L,B,3] 或 [B,3]                       差异特征（可拼接）
    """

    def __init__(self, d,max_iter=30, eps=0.05):
        super().__init__()
        self.d = d
        self.eps = eps
        self.max_iter = max_iter
        # 可学温度（控制门控灵敏度）
        self.tau = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in ['v', 't', 'a']
        })

        # 可学融合权重（协同 vs 差异）
        self.w_coop = nn.Parameter(torch.ones(3) / 3)  # v,t,a
        self.w_disc = nn.Parameter(torch.ones(3) / 3)  # 差异分支权重

        # 输出层
        self.coop_norm = nn.LayerNorm(d)
        self.disc_proj = nn.Linear(3, d)  # 差异特征映射到 d 维（可选）

    def ot_plan(self, X, Y):
        """返回 OT 计划 Π 和距离矩阵 C"""
        # X, Y = X.transpose(0, 1), Y.transpose(0, 1)  # [B,L,d]
        # B, L, d = X.shape
        # C = torch.cdist(X, Y, p=2) ** 2  # [B,L,L]
        # p = q = torch.ones(L, device=X.device) / L
        # Pi = torch.stack([sinkhorn(p, q, C[b], reg=self.eps, numItermax=30)
        #                   for   b in range(B)])  # [B,L,L]
        # return Pi, C
        x = X.permute(1, 0, 2)
        y = Y.permute(1, 0, 2)
        B, L, d = x.shape
        # 2. 归一化 (可选，但在跨模态对齐中强烈建议)
        # 这样可以使用余弦距离，更加稳定
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        # 3. 计算代价矩阵 Cost Matrix (B, L, L)
        # 使用余弦距离: C = 1 - CosineSimilarity
        # x @ y.transpose: (B, L, d) @ (B, d, L) -> (B, L, L)
        sim_matrix = torch.bmm(x, y.transpose(1, 2))
        C = 1 - sim_matrix
        # 4. Sinkhorn 算法求解
        # 初始化边缘分布 (Marginals)，假设每个 token 权重相等 (Uniform)
        # shape: (B, L, 1)
        mu = torch.empty(B, L, 1, device=x.device).fill_(1.0 / L)
        nu = torch.empty(B, L, 1, device=x.device).fill_(1.0 / L)

        # 计算 Kernel K = exp(-C / eps)
        K = torch.exp(-C / self.eps)

        u = torch.zeros_like(mu).fill_(1.0 / L)  # 初始化

        for _ in range(self.max_iter):
            v = nu / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
            u = mu / (torch.bmm(K, v) + 1e-8)

        # 5. 计算传输矩阵 Pi (Transport Plan)
        # Pi = diag(u) @ K @ diag(v)
        # 利用广播机制计算: u * K * v.T (注意维度匹配)
        # u: (B, L, 1), v: (B, L, 1) -> v.transpose: (B, 1, L)
        Pi = u * K * v.transpose(1, 2)  # shape (B, L, L)
        return Pi,C
    def forward(self, G, L_dict):
        lv, lt, la = L_dict['v'], L_dict['t'], L_dict['a']
        L, B, d = lv.shape

        Pi_vt, _ = self.ot_plan(lv, lt)
        Pi_va, _ = self.ot_plan(lv, la)
        Pi_vg,C_vg = self.ot_plan(lv, G)
        # S_v =(Pi_vt @ lt.transpose(0, 1) + Pi_va @ la.transpose(0, 1)+Pi_vg @ G.transpose(0, 1))/3
        S_v =lv.transpose(0, 1)+0.5*Pi_vg.transpose(1,2) @ lv.transpose(0, 1)


        Pi_tv, _ = self.ot_plan(lt, lv)
        Pi_ta, _ = self.ot_plan(lt, la)
        Pi_tg, C_tg = self.ot_plan(lt, G)
        # S_t =  (Pi_tv @ lv.transpose(0, 1) + Pi_ta @ la.transpose(0, 1)+Pi_tg @ G.transpose(0, 1))/3
        S_t = lt.transpose(0, 1)+0.3*Pi_tg.transpose(1,2) @ lt.transpose(0, 1)

        Pi_av, _ = self.ot_plan(la, lv)
        Pi_at, _ = self.ot_plan(la, lt)
        Pi_ag, C_ag = self.ot_plan(la, G)
        # S_a = (Pi_av @ lv.transpose(0, 1) + Pi_at @ lt.transpose(0, 1)+Pi_ag @ G.transpose(0, 1))*0.5

        S_a =la.transpose(0, 1)+0.5*Pi_ag.transpose(1,2) @ la.transpose(0, 1)


        S_dict = {'v': S_v.transpose(0, 1), 't': S_t.transpose(0, 1), 'a': S_a.transpose(0, 1)}
        C_dict={'v': C_vg, 't': C_tg, 'a': C_ag}
        # -------- 2. 全局-局部一致性（协同 + 差异）--------
        D_dict, delta_dict, Gc_dict, coop_dict = {}, {}, {}, {}
        for m in ['v', 't', 'a']:
            Pi, C = self.ot_plan(L_dict[m], G)

            # 一致性细节 D_m
            D_dict[m] = Pi @ G.transpose(0, 1)  # [B,L,d]
            D_dict[m] = D_dict[m].transpose(0, 1)  # [L,B,d]

            # 差异度 δ_m（搬运代价）
            delta = (Pi * C).sum(dim=2)  # [B,L]
            delta_dict[m] = delta.transpose(0, 1)  # [L,B]

            # 门控 Gc_m（置信度）
            tau = self.tau[m].abs() + 1e-4
            Gc = torch.sigmoid((tau - delta) / tau)  # [B,L] -> (0,1)
            Gc_dict[m] = Gc.transpose(0, 1).unsqueeze(-1)  # [L,B,1]

            # 协同表示 C_m = Gc_m ⊙ D_m + (1-Gc_m) ⊙ S_m
            # coop_dict[m] = torch.cat([Gc_dict[m] * D_dict[m],(1 - Gc_dict[m]) * S_dict[m]], dim=0)
            coop_dict[m] = Gc_dict[m] * D_dict[m] + (1 - Gc_dict[m]) * S_dict[m]

        # -------- 3. 融合双分支 --------
        # 3.1 协同分支：加权求和
        w = F.softmax(self.w_coop, dim=0)  # [3]
        coop_stack = torch.stack([coop_dict[m] for m in ['v', 't', 'a']], dim=0)  # [3,L,B,d]
        G_coop = (coop_stack * w.view(3, 1, 1, 1)).sum(0)  # [L,B,d]

        # 3.2 差异分支：拼接 + 线性映射
        delta_stack = torch.stack([delta_dict[m] for m in ['v', 't', 'a']], dim=2)  # [L,B,3]
        disc_feat = self.disc_proj(delta_stack)  # [L,B,d]

        # 3.3 最终融合（残差 + LayerNorm）
        # G_fused = self.coop_norm(G + G_coop + disc_feat)
        G_fused = torch.cat([G_coop, disc_feat], dim=0)

        return {
            'coop_dict': coop_dict,  # 协同表示 {'v':[L,B,d], ...}
            'disc_dict': delta_dict,  # 差异度 {'v':[L,B], ...}
            'D_dict': D_dict,  # 协同表示 {'v':[L,B,d], ...}
            'S_dict': S_dict,  # 差异表示{'v':[L,B,d], ...}
            'C_dict': C_dict,  # 代价矩阵
            'disc_feat': disc_feat,  # 融合差异特征 [L,B,d]
            'G_fused': G_fused,  # 融合后全局 [L,B,d]
            'gating': Gc_dict  # 门控 {'v':[L,B,1], ...}（可视化用）
        }