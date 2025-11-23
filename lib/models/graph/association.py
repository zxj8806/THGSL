import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeScorer(nn.Module):
    def __init__(self, in_dim, hidden=128, geom_dim=5, dropout=0.1):
        super().__init__()
        self.lin  = nn.Linear(in_dim, hidden, bias=False)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden*2 + geom_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        nn.init.xavier_uniform_(self.lin.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _edge_geom(xy, edges):
        if xy.numel() == 0 or edges.numel() == 0:
            return xy.new_zeros((0, 5))
        src, dst = edges[0].long(), edges[1].long()
        xy_s = xy[src]
        xy_d = xy[dst]
        d = xy_d - xy_s
        dx, dy = d[:, 0], d[:, 1]
        dist = torch.clamp(torch.sqrt(dx*dx + dy*dy) + 1e-6, max=1e6)
        ang = torch.atan2(dy, dx + 1e-6)
        geom = torch.stack([dx, dy, dist, torch.cos(ang), torch.sin(ang)], dim=-1)

        scale = max(float(xy[:, 0].max().item() - xy[:, 0].min().item() + 1),
                    float(xy[:, 1].max().item() - xy[:, 1].min().item() + 1))
        scale = max(scale / 8.0, 1.0)
        geom[:, :3] = geom[:, :3] / scale
        return geom

    @staticmethod
    def _edge_geom_with_types(xy, edges, node_types):

        geom = EdgeScorer._edge_geom(xy, edges)
        if node_types is None or geom.shape[0] == 0:
            return geom
        src, dst = edges[0].long(), edges[1].long()
        t_src = node_types[src].float()
        t_dst = node_types[dst].float()
        add = torch.stack([t_src, t_dst], dim=-1)
        return torch.cat([geom, add], dim=-1)

    def forward(self, node_feat, edges, node_xy=None, node_types=None):
        assert node_feat.dim() == 3 and edges.dim() == 3
        B, K, C = node_feat.shape
        _, _, E = edges.shape
        out = []
        for b in range(B):
            if E == 0 or K == 0:
                out.append(node_feat.new_zeros((0,)))
                continue
            x  = node_feat[b]
            e  = edges[b]
            h  = self.lin(x)
            src, dst = e[0].long(), e[1].long()
            hs, hd    = h[src], h[dst]
            if node_xy is not None:
                if node_types is not None:
                    geom = self._edge_geom_with_types(node_xy[b], e, node_types[b]).to(hs.dtype).to(hs.device)  # (E,7)
                else:
                    geom = self._edge_geom(node_xy[b], e).to(hs.dtype).to(hs.device)  # (E,5)
            else:
                gdim = 7 if node_types is not None else 5
                geom = hs.new_zeros((hs.shape[0], gdim))
            z = torch.cat([hs, hd, geom], dim=-1)
            logit = self.mlp(z).squeeze(-1)
            out.append(logit)
        return torch.stack(out, dim=0) if E > 0 else node_feat.new_zeros((B, 0))

@torch.no_grad()
def _boxes_iou_xyxy(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.size(0), b.size(0)))
    x11, y11, x12, y12 = a[:,0], a[:,1], a[:,2], a[:,3]
    x21, y21, x22, y22 = b[:,0], b[:,1], b[:,2], b[:,3]
    xa1 = torch.max(x11[:,None], x21[None,:])
    ya1 = torch.max(y11[:,None], y21[None,:])
    xa2 = torch.min(x12[:,None], x22[None,:])
    ya2 = torch.min(y12[:,None], y22[None,:])
    inter = (xa2 - xa1).clamp(min=0) * (ya2 - ya1).clamp(min=0)
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1[:,None] + area2[None,:] - inter + 1e-6
    return inter / union

@torch.no_grad()
def _assign_nodes_to_gt(centers_xy, gt_boxes, gt_ids, iou_thr=0.3):
    K = centers_xy.shape[0]
    if K == 0 or gt_boxes.numel() == 0:
        return centers_xy.new_full((K,), -1, dtype=torch.long)
    cx, cy = centers_xy[:,0], centers_xy[:,1]
    tiny = torch.stack([cx-0.5, cy-0.5, cx+0.5, cy+0.5], dim=-1).to(gt_boxes.dtype).to(gt_boxes.device)
    iou = _boxes_iou_xyxy(tiny, gt_boxes)
    mval, midx = iou.max(dim=1)
    assigned = gt_ids.new_full((K,), -1)
    assigned[mval >= iou_thr] = gt_ids[midx[mval >= iou_thr]]
    return assigned.long()

@torch.no_grad()
def label_edges_from_gt(graph_pack, gt_boxes_bxn, gt_ids_bxn, t_slices, pos_only_adjacent=True, iou_thr=0.3):
    B, _, E = graph_pack['edges'].shape
    labels = []
    masks  = []
    node_xy   = graph_pack['node_xy']
    node_t    = graph_pack['node_time']
    edges     = graph_pack['edges']

    for b in range(B):
        if E == 0:
            labels.append(edges.new_zeros((0,), dtype=torch.float32))
            masks.append(edges.new_zeros((0,), dtype=torch.float32))
            continue

        t2assign = {}
        for t, (st, ed) in enumerate(t_slices):
            Kt = ed - st
            if Kt <= 0:
                t2assign[t] = node_xy.new_full((0,), -1, dtype=torch.long)
                continue
            centers_t = node_xy[b, st:ed]
            gt_b = gt_boxes_bxn[b, t]
            gt_i = gt_ids_bxn[b, t]
            valid = (gt_i >= 0) & (gt_b[...,0] >= 0)
            gt_b = gt_b[valid]
            gt_i = gt_i[valid].long()
            if gt_b.numel() == 0:
                t2assign[t] = centers_t.new_full((Kt,), -1, dtype=torch.long)
            else:
                t2assign[t] = _assign_nodes_to_gt(centers_t, gt_b, gt_i, iou_thr=iou_thr)

        e = edges[b]
        u, v = e[0].long(), e[1].long()
        tu, tv = node_t[b, u].long(), node_t[b, v].long()
        if pos_only_adjacent:
            mask = (tu - tv).abs() == 1
        else:
            mask = (tu != tv)
        mask = mask.to(torch.float32)

        lab = edges.new_zeros((e.shape[1],), dtype=torch.float32)
        if mask.sum() > 0:
            t2start = {t: st for t,(st,ed) in enumerate(t_slices)}
            u_loc = u - torch.tensor([t2start[int(tt)] for tt in tu.tolist()], device=u.device)
            v_loc = v - torch.tensor([t2start[int(tt)] for tt in tv.tolist()], device=v.device)
            au = torch.tensor([int(t2assign[int(tt)][int(ii)]) if 0 <= int(ii) < len(t2assign[int(tt)]) else -1
                               for tt,ii in zip(tu.tolist(), u_loc.tolist())], device=u.device)
            av = torch.tensor([int(t2assign[int(tt)][int(ii)]) if 0 <= int(ii) < len(t2assign[int(tt)]) else -1
                               for tt,ii in zip(tv.tolist(), v_loc.tolist())], device=v.device)
            lab = (au == av) & (au >= 0) & (av >= 0)
            lab = lab.to(torch.float32)

        labels.append(lab)
        masks.append(mask)

    return torch.stack(labels, dim=0), torch.stack(masks, dim=0)

def assoc_bce_loss(logits, labels, mask, pos_weight=2.0):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    m = mask > 0.5
    if m.sum() == 0:
        return logits.new_tensor(0.0)
    logits_v = logits[m]
    labels_v = labels[m]
    if (labels_v == 1).sum() > 0:
        pw = (labels_v == 0).sum().clamp(min=1).float() / (labels_v == 1).sum().clamp(min=1).float()
        pw = torch.clamp(pw, max=pos_weight)
    else:
        pw = logits_v.new_tensor(pos_weight)
    loss = F.binary_cross_entropy_with_logits(logits_v, labels_v, pos_weight=pw)
    return loss
