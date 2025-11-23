import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LearnableTopoGNN(nn.Module):

    def __init__(self, in_dim, hidden_dim=64, out_dim=None, dropout=0.0,
                 edge_topk=8, temperature=1.0, residual=True):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        self.phi = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.phi.weight)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
        self.in_dim  = in_dim
        self.use_residual = residual and (out_dim == in_dim)
        self.edge_topk = edge_topk
        self.temperature = max(1e-6, float(temperature))

    def _per_dst_softmax(self, logits, dst, K):

        if hasattr(torch.zeros(1), 'scatter_reduce'):
            max_per_dst = torch.full((K,), float('-inf'), device=logits.device)
            max_per_dst = max_per_dst.scatter_reduce(0, dst, logits, reduce='amax', include_self=True)
        else:
            max_per_dst = torch.full((K,), float('-inf'), device=logits.device)
            try:
                max_per_dst = max_per_dst.index_reduce_(0, dst, logits, reduce='amax', include_self=True)
            except Exception:
                for j in torch.unique(dst):
                    jj = (dst == j)
                    if jj.any():
                        max_per_dst[j] = torch.max(logits[jj])

        alpha = torch.exp((logits - max_per_dst[dst]) / self.temperature)
        den = torch.zeros(K, device=logits.device)
        den.index_add_(0, dst, alpha)
        w = alpha / (den[dst] + 1e-6)
        return w

    def _topk_mask_per_dst(self, logits, dst, K, k):

        if k is None or k <= 0:
            return torch.ones_like(logits, dtype=torch.bool)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for j in torch.unique(dst):
            jj = (dst == j).nonzero(as_tuple=False).view(-1)
            if jj.numel() == 0:
                continue
            kk = int(min(k, jj.numel()))
            vals = logits[jj]
            keep_idx = torch.topk(vals, kk, dim=0).indices
            mask[jj[keep_idx]] = True
        return mask

    def forward(self, node_feat, edges, node_xy=None, edge_logits=None, node_time=None, **kwargs):

        assert node_feat.dim() == 3 and edges.dim() == 3
        B, K, C = node_feat.shape
        _, _, E = edges.shape
        if E == 0 or K == 0:
            return node_feat

        out_batches = []
        curr_topk = kwargs.get('edge_topk', None)
        tau = kwargs.get('temperature', None)
        if curr_topk is None:
            try:
                curr_topk = int(os.environ.get('GSL_TOPK', str(self.edge_topk)))
            except Exception:
                curr_topk = self.edge_topk
        if tau is None:
            try:
                tau = float(os.environ.get('GSL_TAU', str(self.temperature)))
            except Exception:
                tau = self.temperature

        for b in range(B):

            causal = (os.environ.get('GSL_CAUSAL', '1') == '1')
            allow_intra = (os.environ.get('GSL_ALLOW_INTRA', '1') == '1')
            allow_backward = (os.environ.get('GSL_ALLOW_BACKWARD', '0') == '1')
            e = edges[b]
            src = e[0].long()
            dst = e[1].long()
            if edge_logits is None:
                logits_b = torch.zeros(src.numel(), device=node_feat.device, dtype=node_feat.dtype)
            else:
                logits_b = edge_logits[b].to(node_feat.dtype)

            topk_mask = self._topk_mask_per_dst(logits_b.detach(), dst, K, curr_topk)
            masked_logits = logits_b.masked_fill(~topk_mask, -1e9)

            if node_time is not None and e.numel() > 0 and causal:
                t_b = node_time[b].long()
                src_t = t_b[src]
                dst_t = t_b[dst]
                valid_mask = (dst_t > src_t)
                if allow_intra:
                    valid_mask = valid_mask | (dst_t == src_t)
                if allow_backward:
                    valid_mask = valid_mask | (dst_t < src_t)
            else:
                valid_mask = torch.ones_like(logits_b, dtype=torch.bool)
            
            logits_masked_temporal = logits_b.masked_fill(~valid_mask, -1e9)
            topk_mask = self._topk_mask_per_dst(logits_masked_temporal.detach(), dst, K, curr_topk)
            masked_logits = logits_masked_temporal.masked_fill(~topk_mask, -1e9)
            weights = self._per_dst_softmax(masked_logits, dst, K)

            x = node_feat[b]
            msg_src = self.phi(x)[src]
            msg_src = self.dropout(msg_src * weights.unsqueeze(-1))

            x_new = torch.zeros(K, self.out_dim, device=x.device, dtype=msg_src.dtype)
            idx2 = dst.unsqueeze(-1).expand_as(msg_src)
            x_new.scatter_add_(0, idx2, msg_src)

            if self.use_residual and x_new.shape[-1] == x.shape[-1]:
                x_new = x_new + x

            out_batches.append(x_new)

        return torch.stack(out_batches, dim=0)
