from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.utils.data_parallel import DataParallel
from lib.utils.utils import AverageMeter
from lib.utils.decode import trainer_decode
from lib.utils.post_process import ctdet_post_process
import numpy as np


def _to_tensor(x):
    import torch
    if torch.is_tensor(x):
        return x
    try:
        return torch.tensor(x, dtype=torch.float32)
    except Exception:
        try:
            return torch.tensor(0.0 if x is None else float(x), dtype=torch.float32)
        except Exception:
            return torch.tensor(0.0, dtype=torch.float32)


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=True)
    if isinstance(x, dict):
        return {kk: _to_device(vv, device) for kk, vv in x.items()}
    if isinstance(x, (list, tuple)):
        xs = [_to_device(vv, device) for vv in x]
        return tuple(xs) if isinstance(x, tuple) else xs
    return x

def post_process(output, meta, num_classes, scale=1):

    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']

    torch.cuda.synchronize()

    decoded = trainer_decode(hm, wh, reg=reg)

    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]

    if isinstance(decoded, dict) and 'bboxes' not in decoded:
        results = {}
        for k, v in decoded.items():
            try:
                cls_id = int(k)
            except Exception:
                continue
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 1 and v.size == 5:
                v = v.reshape(1, 5)
            if v.ndim == 2 and v.shape[1] >= 5:
                arr = v[:, :5].astype(np.float32)
            else:
                continue
            if scale != 1:
                arr[:, :4] /= float(scale)
            results[cls_id] = arr
        return results

    if isinstance(decoded, dict) and (
        'bboxes' in decoded and ('clses' in decoded or 'classes' in decoded or 'labels' in decoded) and 'scores' in decoded
    ):
        b = decoded['bboxes']
        s = decoded['scores']
        c = decoded.get('clses', decoded.get('classes', decoded.get('labels')))

        if torch.is_tensor(b): b = b.detach().cpu().numpy()
        if torch.is_tensor(s): s = s.detach().cpu().numpy()
        if torch.is_tensor(c): c = c.detach().cpu().numpy()

        b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        c = np.asarray(c).astype(np.int32).reshape(-1)

        if c.size > 0 and c.min() == 0:
            c = c + 1

        results = {}
        for cls_id in np.unique(c):
            mask = (c == cls_id)
            if not np.any(mask):
                continue
            boxes = b[mask]
            scores = s[mask].reshape(-1, 1)
            arr = np.concatenate([boxes, scores], axis=1).astype(np.float32)
            if scale != 1:
                arr[:, :4] /= float(scale)
            results[int(cls_id)] = arr
        return results

    dets = decoded.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]


def merge_outputs(detections, num_classes, max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = 3000
        end = time.time()
        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            batch = {
                k: (v if k in ('meta', 'file_name') else _to_device(v, opt.device))
                for k, v in batch.items()
            }

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)

            print('phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f, track_loss=%0.4f'
                  % (phase, epoch, iter_id + 1, num_iters, time.time() - end,
                     loss.mean().cpu().detach().numpy(),
                     _to_tensor(loss_stats['hm_loss']).mean().cpu().detach().numpy(),
                     _to_tensor(loss_stats['wh_loss']).mean().cpu().detach().numpy(),
                     _to_tensor(loss_stats['off_loss']).mean().cpu().detach().numpy(),
                     _to_tensor(loss_stats['track_loss']).mean().cpu().detach().numpy()))

            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    _to_tensor(loss_stats[l]).mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = 1 / 60.

        return ret, results


    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
