import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['AlignSubNet']


class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        From: https://github.com/yaohungt/Multimodal-Transformer
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len + 1, num_layers=2,
                                                        batch_first=True)  # 1 denoting blank

        self.out_seq_len = out_seq_len

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank.
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(
            pred_output_position_inclu_blank)  # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :,
                                    1:]  # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1, 2)  # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x)  # batch_size x out_seq_len x in_dim

        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        # return pseudo_aligned_out, (pred_output_position_inclu_blank)
        return pseudo_aligned_out


class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        """
        mode: the way of aligning
            avg_pool, ctc, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d']

        in_dim_t, in_dim_a, in_dim_v = args.feature_dims
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens
        self.dst_len = seq_len_t
        self.mode = mode

        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool,
            'ctc': self.__ctc,
            'conv1d': self.__conv1d
        }

        if mode == 'conv1d':
            self.conv1d_T = nn.Conv1d(seq_len_t, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_A = nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_V = nn.Conv1d(seq_len_v, self.dst_len, kernel_size=1, bias=False)
        elif mode == 'ctc':
            self.ctc_t = CTCModule(in_dim_t, self.dst_len)
            self.ctc_a = CTCModule(in_dim_a, self.dst_len)
            self.ctc_v = CTCModule(in_dim_v, self.dst_len)

    def get_seq_len(self):
        return self.dst_len

    def __ctc(self, text_x, audio_x, video_x):
        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        return text_x, audio_x, video_x

    def __avg_pool(self, text_x, audio_x, video_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x

        text_x = align(text_x)
        audio_x = align(audio_x)
        video_x = align(video_x)
        return text_x, audio_x, video_x

    def __conv1d(self, text_x, audio_x, video_x):
        text_x = self.conv1d_T(text_x) if text_x.size(1) != self.dst_len else text_x
        audio_x = self.conv1d_A(text_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.conv1d_V(text_x) if video_x.size(1) != self.dst_len else video_x
        return text_x, audio_x, video_x

    def forward(self, text_x, audio_x, video_x):
        # already aligned
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x
        return self.ALIGN_WAY[self.mode](text_x, audio_x, video_x)


class CMDLoss(nn.Module):
    """
    Central Moment Discrepancy Loss
    参数:
        k_moments: 使用到多少阶矩（推荐 3~5）
    """
    def __init__(self, k_moments=5):
        super(CMDLoss, self).__init__()
        self.k = k_moments

    # ---------- 单阶中心矩 ----------
    def _moment(self, x, k):
        # x: [B, d]
        mean = x.mean(dim=0, keepdim=True)          # [1, d]
        delta = x - mean                            # [B, d]
        return torch.mean(delta ** k, dim=0)        # [d]

    # ---------- CMD ----------
    def forward(self, x, y):
        """
        x, y: [B, d] 两个模态的 batch 特征
        返回: 标量 loss
        """
        loss = 0.0
        for k in range(1, self.k + 1):
            mk_x = self._moment(x, k)
            mk_y = self._moment(y, k)
            loss += torch.norm(mk_x - mk_y, p=2)   # L2 距离
        return loss

def uni_distill(logits1, logits2):
    prob1 = torch.softmax(logits1, dim=-1)
    prob2 = torch.softmax(logits2, dim=-1)
    mse = torch.mean((prob1 - prob2) ** 2, dim=-1)
    return torch.mean(mse)


class TCA(torch.nn.Module):
    """
        Token-level Cross-modal Alignment
        以文本 Rt 为 anchor，分别与视觉 Rv、音频 Ra 做 token-wise 对比学习。
        """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, Rt: torch.Tensor, Rv: torch.Tensor, Ra: torch.Tensor):
        """
        Rt/Rv/Ra: (B, n, d)  float
        return: loss_vt, loss_at
        """
        B, n, d = Rt.shape
        device = Rt.device

        # ---------- 视觉-文本 ----------
        # step-1 token-wise similarity  (B, n, n)
        S_vt = torch.bmm(Rv, Rt.transpose(1, 2)) / self.tau  # (B,n,n)

        # step-2  token-to-sentence  (B,n)
        w_vt = F.softmax(S_vt, dim=-1)  # 对文本维度归一化
        sent_vt = (w_vt * S_vt).sum(dim=-1)  # (B,n)

        # step-3  sentence-level  (B,)
        w_sent = F.softmax(sent_vt, dim=-1)  # 对视觉token归一化
        score_vt = (w_sent * sent_vt).sum(dim=-1)  # (B,)

        # ---------- 音频-文本 同理 ----------
        S_at = torch.bmm(Ra, Rt.transpose(1, 2)) / self.tau
        w_at = F.softmax(S_at, dim=-1)
        sent_at = (w_at * S_at).sum(dim=-1)
        w_sent_a = F.softmax(sent_at, dim=-1)
        score_at = (w_sent_a * sent_at).sum(dim=-1)

        # ---------- 视频-音频 同理 ----------
        S_va = torch.bmm(Rv, Ra.transpose(1, 2)) / self.tau
        w_va = F.softmax(S_va, dim=-1)
        sent_va = (w_va * S_va).sum(dim=-1)
        w_sent_va = F.softmax(sent_va, dim=-1)
        score_va = (w_sent_va * sent_va).sum(dim=-1)

        # ---------- InfoNCE ----------
        def infonce(score):
            # score: (B,)  正例对角线
            logits = score.unsqueeze(1)  # (1,B)
            logits = logits.expand(-1, score.size(0))
            labels = torch.arange(score.size(0), device=score.device)
            # print(logits.shape, labels.shape)
            return F.cross_entropy(logits, labels)

        loss_vt = infonce(score_vt)
        loss_at = infonce(score_at)
        loss_va=infonce(score_va)
        return loss_vt, loss_at,loss_va


# if __name__ == "__main__":
#     B, n, d = 4, 8, 128
#     Rt = torch.randn(B, n, d).cuda()
#     Rv = torch.randn(B, n, d).cuda()
#     Ra = torch.randn(B, n, d).cuda()
#
#     tca = TCA(temperature=0.1).cuda()
#     lvt, lat = tca(Rt, Rv, Ra)
#     print("loss_vt:", lvt.item(), "loss_at:", lat.item())

class router(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim / 8))
        self.l2 = nn.Linear(int(dim / 8), channel_num)
        self.t = t

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1))) / self.t
        output = torch.softmax(x, dim=1)
        return output