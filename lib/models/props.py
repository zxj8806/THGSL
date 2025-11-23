import torch
import torch.nn.functional as F

@torch.no_grad()
def _topk_from_heatmap(hm, K=50):

    B, C, H, W = hm.shape
    hm = torch.sigmoid(hm)
    hm = hm.view(B, C, -1)
    scores_per_cls, inds_per_cls = torch.topk(hm, K, dim=2)

    scores_flat = scores_per_cls.view(B, -1)
    inds_flat   = inds_per_cls.view(B, -1)

    clses = torch.arange(C, device=hm.device).view(1, C, 1).expand(B, C, K).contiguous().view(B, -1)
    final_scores, topk_idx = torch.topk(scores_flat, K, dim=1)
    final_inds  = inds_flat.gather(1, topk_idx)
    final_clses = clses.gather(1, topk_idx)

    ys = (final_inds // W).float()
    xs = (final_inds %  W).float()
    return final_scores, final_clses.long(), xs, ys, final_inds.long()

@torch.no_grad()
def _gather_feat(feat, ind):

    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
    ind = ind.unsqueeze(-1).expand(-1, -1, C)
    gathered = feat.gather(1, ind)
    return gathered

@torch.no_grad()
def _sample_points_from_feature(feat, xs, ys):

    B, C, H, W = feat.shape
    x_norm = (xs / (W - 1) * 2) - 1.0
    y_norm = (ys / (H - 1) * 2) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(feat, grid, mode='bilinear', align_corners=True)
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
    return sampled

@torch.no_grad()
def build_proposals_lastframe(
    feat_last,
    hm, wh, reg,
    topk=50
):

    B, Ccls, H, W = hm.shape
    scores, clses, xs, ys, inds = _topk_from_heatmap(hm, K=topk)

    reg_at = _gather_feat(reg, inds)
    wh_at  = _gather_feat(wh,  inds)

    xs = xs + reg_at[..., 0]
    ys = ys + reg_at[..., 1]

    half_w = wh_at[..., 0] / 2.0
    half_h = wh_at[..., 1] / 2.0
    x1 = xs - half_w
    y1 = ys - half_h
    x2 = xs + half_w
    y2 = ys + half_h
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    feats = _sample_points_from_feature(feat_last, xs, ys)

    props = {
        'scores':  scores.detach(),
        'classes': clses.detach(),
        'centers': torch.stack([xs, ys], dim=-1).detach(),
        'boxes':   boxes.detach(),
        'feats':   feats,
        'hw': torch.tensor([H, W], device=feat_last.device),
    }
    return props
