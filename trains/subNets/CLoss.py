import torch
import torch.nn as nn
import torch.nn.functional as F


class HCL(nn.Module):
    """
    第二层次: 融合模态-单模态对比学习
    输入: 融合特征 [L, B, d], 单模态特征 [L, B, d]
    """

    def __init__(self, input_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        # 1D卷积: 处理时序并投影到对比空间
        # 输入 [B, d, L] -> 输出 [B, proj_dim, L]
        self.conv_fusion = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.conv_unimodal = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        # MFB分布估计器 (用于KL约束)
        self.mu_estimator = nn.Linear(input_dim, input_dim)
        self.sigma_estimator = nn.Linear(input_dim, input_dim)
        # 全局池化: 聚合时间维度
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, fusion_seq, unimodal_seq):
        """
        Args:
            fusion_seq:   [L, Batch, Dim]
            unimodal_seq: [L, Batch, Dim]
        """
        # 1. 调整维度适配Conv1d: [L,B,d] -> [B,d,L]
        fusion = fusion_seq.permute(1, 2, 0)  # [B, d, L]
        unimodal = unimodal_seq.permute(1, 2, 0)  # [B, d, L]

        # 2. 投影到统一空间并提取时序模式
        fusion_proj = self.conv_fusion(fusion)  # [B, proj_dim, L]
        unimodal_proj = self.conv_unimodal(unimodal)  # [B, proj_dim, L]

        # 3. 全局池化 -> 向量表示 [B, proj_dim]
        fusion_vec = self.global_pool(fusion_proj).squeeze(-1)
        unimodal_vec = self.global_pool(unimodal_proj).squeeze(-1)
        #kl_loss
        mu = self.mu_estimator(fusion_vec)  # [B, d]
        sigma = F.softplus(self.sigma_estimator(fusion_vec))  # [B, d]
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2, dim=1).mean()

        # 4. L2归一化 (对比学习标准)
        fusion_vec = F.normalize(fusion_vec, dim=1)
        unimodal_vec = F.normalize(unimodal_vec, dim=1)

        # 5. 计算相似度矩阵 [B, B]
        sim_matrix = torch.matmul(fusion_vec, unimodal_vec.T) / self.temperature

        # 6. 正样本标签 (对角线)
        batch_size = fusion_vec.size(0)
        labels = torch.arange(batch_size, device=fusion_vec.device)


        # 7. 双向InfoNCE损失
        loss_f2u = F.cross_entropy(sim_matrix, labels)  # 融合->单模态
        # loss_u2f = F.cross_entropy(sim_matrix.T, labels)  # 单模态->融合
        #
        # return (loss_f2u + loss_u2f) / 2
        return loss_f2u


# 使用示例
# L, B, d = 50, 32, 128
# fusion = torch.randn(L, B, d, requires_grad=True)
# unimodal = torch.randn(L, B, d, requires_grad=True)
#
# criterion = HCL(input_dim=d)
# loss = criterion(fusion, unimodal)  # 标量损失
# loss.backward()