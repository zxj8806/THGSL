import torch

class GraphBuilder:
    def __init__(self, knn_k=8, use_knn=True):
        self.knn_k = knn_k
        self.use_knn = use_knn

    @torch.no_grad()
    def _pairwise_dist2(self, xy):
        dif = xy[:, None, :] - xy[None, :, :]
        return (dif * dif).sum(-1)

    @torch.no_grad()
    def build(self, proposals):
        feats = proposals['feats']
        centers = proposals['centers']
        assert feats.dim() == 3 and centers.dim() == 3
        B, K, C = feats.shape

        node_feat = feats.clone()
        node_xy = centers.clone()

        edges_list = []
        for b in range(B):
            xy = centers[b]
            if self.use_knn and K > 1:
                d2 = self._pairwise_dist2(xy)
                d2 = d2 + torch.eye(K, device=xy.device) * 1e9
                knn_k = min(self.knn_k, max(1, K-1))
                idx = torch.topk(-d2, k=knn_k, dim=1).indices
                src = torch.arange(K, device=xy.device).unsqueeze(1).expand_as(idx)
                e1 = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=0)
                e2 = torch.stack([e1[1], e1[0]], dim=0)
                edges = torch.cat([e1, e2], dim=1)
            else:
                ii, jj = torch.where(torch.ones(K, K, device=xy.device) - torch.eye(K, device=xy.device) > 0)
                edges = torch.stack([ii, jj], dim=0)

            edges_list.append(edges.unsqueeze(0))

        edges = torch.cat(edges_list, dim=0)
        return {
            'node_feat': node_feat,
            'node_xy': node_xy,
            'edges': edges,
        }

class TemporalGraphBuilder:

    def __init__(self, knn_intra=8, knn_inter=4, use_knn=True,
                 align=True, align_ema=0.0, align_max_shift=32, align_topk=30):
        import os
        self.knn_intra = knn_intra
        self.knn_inter = knn_inter
        self.use_knn = use_knn
        self.align = align
        self.align_ema = align_ema
        self.align_max_shift = align_max_shift
        self.align_topk = align_topk

        self.allow_pp   = (os.environ.get('HET_ALLOW_PP', '0') == '1')
        self.k_intra_oo = int(os.environ.get('HET_K_INTRA_OO', '8'))
        self.k_intra_op = int(os.environ.get('HET_K_INTRA_OP', '4'))
        self.k_inter_oo = int(os.environ.get('HET_K_INTER_OO', '4'))
        self.k_inter_op = int(os.environ.get('HET_K_INTER_OP', '4'))

    @torch.no_grad()
    def _pairwise_dist2(self, xy):
        dif = xy[:, None, :] - xy[None, :, :]
        return (dif * dif).sum(-1)

    @torch.no_grad()
    def _knn_edges_subset(self, xy, idx_from, idx_to, k):
        device = xy.device
        if idx_from.numel() == 0 or idx_to.numel() == 0 or k <= 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        X = xy[idx_from]
        Y = xy[idx_to]
        pd2 = (X[:,None,:] - Y[None,:,:])
        pd2 = (pd2 * pd2).sum(-1)
        kk = min(k, max(1, Y.shape[0]))
        nbr = torch.topk(-pd2, k=kk, dim=1).indices
        src = idx_from.unsqueeze(1).expand_as(nbr).reshape(-1)
        dst = idx_to[nbr.reshape(-1)]
        return torch.stack([src, dst], dim=0)

    @torch.no_grad()
    def _knn_cross_frames(self, xy_prev, idx_prev, xy_cur, idx_cur, k):
        device = xy_prev.device
        if idx_prev.numel() == 0 or idx_cur.numel() == 0 or k <= 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        X = xy_prev
        Y = xy_cur
        pd2 = (X[:,None,:] - Y[None,:,:])
        pd2 = (pd2 * pd2).sum(-1)
        kk = min(k, max(1, Y.shape[0]))
        nbr = torch.topk(-pd2, k=kk, dim=1).indices
        src = idx_prev.unsqueeze(1).expand_as(nbr).reshape(-1)
        dst = idx_cur[nbr.reshape(-1)]
        return torch.stack([src, dst], dim=0)

    @torch.no_grad()
    def build_seq(self, props_list):
        assert len(props_list) >= 1
        B = props_list[0]['feats'].shape[0]
        feats_cat, xy_cat, types_cat, counts = [], [], [], []
        t_slices = []
        st = 0
        for p in props_list:
            feats_cat.append(p['feats'])
            xy_cat.append(p['centers'])
            if 'types' in p:
                types_cat.append(p['types'].long())
            else:
                types_cat.append(p['feats'].new_zeros((B, p['feats'].shape[1])).long())
            Ki = p['feats'].shape[1]
            counts.append(Ki)
            t_slices.append((st, st + Ki))
            st += Ki

        node_feat = torch.cat(feats_cat, dim=1)
        node_xy   = torch.cat(xy_cat,   dim=1)
        node_types= torch.cat(types_cat,dim=1).long()
        Ksum = node_feat.shape[1]

        node_time = []
        for t, Ki in enumerate(counts):
            node_time.append(torch.full((Ki,), t, dtype=torch.long))
        node_time = torch.cat(node_time, dim=0).unsqueeze(0).expand(B, -1).contiguous()

        edges_b = []
        for b in range(B):
            e_list = []
            for ti, Ki in enumerate(counts):
                if Ki == 0: 
                    continue
                st_i, ed_i = t_slices[ti]
                xy = node_xy[b, st_i:ed_i]
                tp = node_types[b, st_i:ed_i]
                idx_obj = torch.nonzero(tp == 0, as_tuple=False).view(-1) + st_i
                idx_pri = torch.nonzero(tp == 1, as_tuple=False).view(-1) + st_i
                if idx_obj.numel() > 0 and self.k_intra_oo > 0:
                    e_oo = self._knn_edges_subset(node_xy[b], idx_obj, idx_obj, self.k_intra_oo)
                    e_list.append(e_oo)
                if idx_obj.numel() > 0 and idx_pri.numel() > 0 and self.k_intra_op > 0:
                    e_op = self._knn_edges_subset(node_xy[b], idx_obj, idx_pri, self.k_intra_op)
                    e_po = self._knn_edges_subset(node_xy[b], idx_pri, idx_obj, self.k_intra_op)
                    e_list += [e_op, e_po]
                if self.allow_pp and idx_pri.numel() > 1:
                    e_pp = self._knn_edges_subset(node_xy[b], idx_pri, idx_pri, max(1, self.k_intra_op//2))
                    e_list.append(e_pp)
                    
            for ti in range(len(counts) - 1):
                st1, ed1 = t_slices[ti]
                st2, ed2 = t_slices[ti+1]
                if ed1 - st1 == 0 or ed2 - st2 == 0:
                    continue
                tp1 = node_types[b, st1:ed1]
                tp2 = node_types[b, st2:ed2]
                i1o = torch.nonzero(tp1 == 0, as_tuple=False).view(-1) + st1
                i1p = torch.nonzero(tp1 == 1, as_tuple=False).view(-1) + st1
                i2o = torch.nonzero(tp2 == 0, as_tuple=False).view(-1) + st2
                i2p = torch.nonzero(tp2 == 1, as_tuple=False).view(-1) + st2
                if i1o.numel() > 0 and i2o.numel() > 0 and self.k_inter_oo > 0:
                    e_oo_12 = self._knn_cross_frames(node_xy[b][i1o], i1o, node_xy[b][i2o], i2o, self.k_inter_oo)
                    e_oo_21 = self._knn_cross_frames(node_xy[b][i2o], i2o, node_xy[b][i1o], i1o, self.k_inter_oo)
                    e_list += [e_oo_12, e_oo_21]
                if i1o.numel() > 0 and i2p.numel() > 0 and self.k_inter_op > 0:
                    e_op_12 = self._knn_cross_frames(node_xy[b][i1o], i1o, node_xy[b][i2p], i2p, self.k_inter_op)
                    e_list.append(e_op_12)
                if i1p.numel() > 0 and i2o.numel() > 0 and self.k_inter_op > 0:
                    e_po_12 = self._knn_cross_frames(node_xy[b][i1p], i1p, node_xy[b][i2o], i2o, self.k_inter_op)
                    e_list.append(e_po_12)
                if self.allow_pp and i1p.numel() > 0 and i2p.numel() > 0 and self.k_inter_op > 0:
                    e_pp_12 = self._knn_cross_frames(node_xy[b][i1p], i1p, node_xy[b][i2p], i2p, max(1, self.k_inter_op//2))
                    e_list.append(e_pp_12)

            if len(e_list) == 0:
                edges_b.append(torch.zeros(2, 0, dtype=torch.long, device=node_xy.device).unsqueeze(0))
            else:
                edges_b.append(torch.cat(e_list, dim=1).unsqueeze(0))

        edges = torch.cat(edges_b, dim=0)
        return {
            'node_feat': node_feat,
            'node_xy': node_xy,
            'edges': edges,
            'node_time': node_time,
            't_slices': t_slices,
            'node_types': node_types
        }
