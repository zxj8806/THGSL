
from __future__ import absolute_import, division, print_function
import os, random
import numpy as np
import torch
import os

from lib.utils.opts import opts
from lib.dataset.coco_icpr import COCO
from lib.models.main import get_backbone, load_model, save_model
from lib.Trainer.trainer import Trainer

def set_seed(seed=317):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def main(opt):
    set_seed(317)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if len(opt.gpus) > 0 and opt.gpus[0] >= 0 else 'cpu')

    print('==> initializing {} val data.'.format(opt.datasetname))
    DataVal = COCO(opt, 'val')
    val_loader = torch.utils.data.DataLoader(
        DataVal,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print('==> initializing {} train data.'.format(opt.datasetname))
    DataTrain = COCO(opt, 'train')
    train_loader = torch.utils.data.DataLoader(
        DataTrain,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    base_s = DataTrain.coco

    print('Creating model...')
    heads = {
        'hm': opt.num_classes,
        'wh': 2,
        'reg': 2,
        'dis': 2
    }
    model = get_backbone(heads, opt.model_name)
    print(opt.model_name)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    ensure_dir(opt.save_dir)
    ensure_dir(opt.save_results_dir)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')
    best_ap50 = -1
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        freeze_n = int(os.environ.get('GATE_FREEZE_EPOCHS', '0'))
        if 'freeze_n' in locals() or 'freeze_n' in globals():
            try:
                if hasattr(model, 'g2m_gate'):
                    model.g2m_gate.requires_grad = (epoch >= freeze_n)
            except Exception:
                pass
            log_dict_train, _ = trainer.train(epoch, train_loader)

            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)

            if epoch in opt.lr_step:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('==> Drop LR to {:.6f} at epoch {:d}'.format(lr, epoch))


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
