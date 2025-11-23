import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph2MapFusion(nn.Module):

    def __init__(self, in_channels=16, out_channels=16, splat_sigma=0.0):
        super().__init__()
        self.proj_obj   = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.proj_prior = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_obj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_prior.weight, mode='fan_out', nonlinearity='relu')

        self.splat_sigma = splat_sigma
        self.alpha_prior = float(os.environ.get('HET_ALPHA_PRIOR', '1.0'))

        if splat_sigma > 0:
            k = torch.tensor([[0.075, 0.125, 0.075],
                              [0.125, 0.300, 0.125],
                              [0.075, 0.125, 0.075]], dtype=torch.float32)
            self.register_buffer('_kern', k.unsqueeze(0).unsqueeze(0))

    @torch.no_grad()
    def _splat3x3(self, x):
        if not hasattr(self, '_kern'):
            return x
        B, C, H, W = x.shape
        w = self._kern.expand(C, 1, 3, 3).contiguous()
        return F.conv2d(x, w, bias=None, stride=1, padding=1, groups=C)

    @torch.no_grad()
    def rasterize(self, node_feat, node_xy, H, W, mask_index=None):

        B, K, C = node_feat.shape
        device = node_feat.device
        out = node_feat.new_zeros((B, C, H, W))
        if K == 0:
            return out

        if mask_index is not None:
            mask_index = mask_index.to(node_feat.device)
            if mask_index.dtype != torch.bool:
                mask_index = mask_index > 0.5
            if (~mask_index).all():
                return out
            node_feat = node_feat[:, mask_index, :]
            node_xy   = node_xy[:,   mask_index, :]

            K = node_feat.shape[1]
            if K == 0:
                return out

        xs = node_xy[..., 0].round().clamp(0, W-1).long()
        ys = node_xy[..., 1].round().clamp(0, H-1).long()
        idx_flat = (ys * W + xs)

        for b in range(B):
            if K == 0:
                continue
            src = node_feat[b].transpose(0,1).contiguous()
            idx = idx_flat[b].view(1, -1).expand(src.size(0), -1)
            buf = out[b].view(src.size(0), -1)
            buf.scatter_add_(1, idx, src)

        return out

    def forward(self, node_feat, node_xy, hw, node_types=None):
        H = int(hw[0].item() if hasattr(hw[0], 'item') else hw[0])
        W = int(hw[1].item() if hasattr(hw[1], 'item') else hw[1])

        if node_types is None:
            base_map = self.rasterize(node_feat, node_xy, H, W)
            if self.splat_sigma and self.splat_sigma > 0:
                base_map = self._splat3x3(base_map)
            return self.proj_obj(base_map)

        if node_types.dim() == 1:
            node_types = node_types.unsqueeze(0).expand(node_feat.size(0), -1)
        node_types = node_types.long()
        if not hasattr(self, '_het_printed'):
            try:
                B, K = node_types.size()
                obj_cnt = int((node_types==0).sum().item())
                prior_cnt = int((node_types==1).sum().item())
                print(f"[HET] node_types present: obj={obj_cnt}, prior={prior_cnt}, alpha_prior={self.alpha_prior}")
                self._het_printed = True
            except Exception as _:
                pass

        mask_obj   = (node_types == 0)
        mask_prior = (node_types == 1)

        map_obj   = self.rasterize(node_feat, node_xy, H, W, mask_index=mask_obj[0] if mask_obj.size(0)==1 else None)
        map_prior = self.rasterize(node_feat, node_xy, H, W, mask_index=mask_prior[0] if mask_prior.size(0)==1 else None)

        if node_feat.size(0) > 1:
            maps_o, maps_p = [], []
            for b in range(node_feat.size(0)):
                mo = self.rasterize(node_feat[b:b+1], node_xy[b:b+1], H, W, mask_index=mask_obj[b])
                mp = self.rasterize(node_feat[b:b+1], node_xy[b:b+1], H, W, mask_index=mask_prior[b])
                maps_o.append(mo); maps_p.append(mp)
            map_obj   = torch.cat(maps_o, dim=0)
            map_prior = torch.cat(maps_p, dim=0)

        if self.splat_sigma and self.splat_sigma > 0:
            map_obj   = self._splat3x3(map_obj)
            map_prior = self._splat3x3(map_prior)

        out = self.proj_obj(map_obj) + (self.alpha_prior * self.proj_prior(map_prior))
        if not hasattr(self, '_het_printed_map'):
            try:
                m0 = float(map_obj.abs().mean().item())
                m1 = float(map_prior.abs().mean().item())
                print(f"[HET] map stats: obj|map|={m0:.6f}, prior|map|={m1:.6f}")
                self._het_printed_map = True
            except Exception:
                pass
        return out
