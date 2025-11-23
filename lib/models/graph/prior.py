import os
import torch
import torch.nn.functional as F

class GridPriorBuilder:

    def __init__(self, grid=None, max_nodes=None, use_hm=None, hm_thr=None):
        self.grid = int(os.environ.get('PRIOR_GRID', '64') if grid is None else grid)
        self.max_nodes = int(os.environ.get('PRIOR_MAX', '128') if max_nodes is None else max_nodes)
        self.use_hm = (os.environ.get('PRIOR_USE_HM', '1') == '1') if use_hm is None else bool(use_hm)
        self.hm_thr = float(os.environ.get('PRIOR_HM_THR', '0.05') if hm_thr is None else hm_thr)

    @torch.no_grad()
    def _make_grid_centers(self, H, W, device):
        step = max(4, int(self.grid))
        ys = torch.arange(step//2, H, step, device=device)
        xs = torch.arange(step//2, W, step, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([xx.reshape(-1).float(), yy.reshape(-1).float()], dim=-1)
        return centers

    @torch.no_grad()
    def _sample_feat(self, p0, centers_xy):
        B, C, H, W = p0.shape
        if centers_xy.numel() == 0:
            return p0.new_zeros((B, 0, C))
        xs, ys = centers_xy[:,0], centers_xy[:,1]
        gx = (xs / (W - 1) * 2 - 1).clamp(-1, 1)
        gy = (ys / (H - 1) * 2 - 1).clamp(-1, 1)
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(p0, grid, mode='bilinear', align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def build_from_p0(self, p0, hm=None):

        assert p0.dim() == 4
        B, C, H, W = p0.shape
        device = p0.device

        base_centers = self._make_grid_centers(H, W, device)
        K0 = base_centers.shape[0]
        if K0 == 0:
            return {'feats': p0.new_zeros((B,0,C)), 'centers': p0.new_zeros((B,0,2))}

        keep = None
        if self.use_hm and (hm is not None) and (hm.numel() > 0):
            prob = torch.sigmoid(hm.mean(dim=0, keepdim=True))
            xs, ys = base_centers[:,0], base_centers[:,1]
            gx = (xs / (W - 1) * 2 - 1).clamp(-1, 1)
            gy = (ys / (H - 1) * 2 - 1).clamp(-1, 1)
            grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
            prob_k = F.grid_sample(prob, grid, mode='bilinear', align_corners=True).squeeze().view(-1)
            keep = (prob_k > self.hm_thr).nonzero(as_tuple=False).view(-1)

        if keep is None or keep.numel() == 0:
            keep = torch.arange(K0, device=device)

        if keep.numel() > self.max_nodes:
            keep = keep[:self.max_nodes]
        centers = base_centers[keep]
        feats = self._sample_feat(p0, centers)
        centers = centers.unsqueeze(0).expand(B, -1, -1).contiguous()

        return {'feats': feats, 'centers': centers}
