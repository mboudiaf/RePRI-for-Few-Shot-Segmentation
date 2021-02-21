import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from visdom_logger import VisdomLogger
from .model.pspnet import get_model
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_train_loader, get_val_loader
from .util import intersectionAndUnionGPU, get_model_dir, AverageMeter, find_free_port
from .util import setup, cleanup, main_process
from tqdm import tqdm
from .test import standard_validate, episodic_validate
from typing import Dict
from torch import Tensor

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from typing import Tuple
from .util import load_cfg_from_cfg_file, merge_cfg_from_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:

    print(f"==> Running process rank {rank}.")
    setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    callback = None if args.visdom_port == -1 else VisdomLogger(port=args.visdom_port)

    # ========== Model + Optimizer ==========
    model = get_model(args).to(rank)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.bottleneck, model.classifier]

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
    optimizer = get_optimizer(args, params_list)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    savedir = get_model_dir(args)

    # ========== Validation ==================
    validate_fn = episodic_validate if args.episodic_val else standard_validate

    # ========== Data  =====================
    train_loader, train_sampler = get_train_loader(args)
    val_loader, _ = get_val_loader(args)  # mode='train' means that we will validate on images from validation set, but with the bases classes

    # ========== Scheduler  ================
    scheduler = get_scheduler(args, optimizer, len(train_loader))

    # ========== Metrics initialization ====
    max_val_mIoU = 0.
    if args.debug:
        iter_per_epoch = 5
    else:
        iter_per_epoch = len(train_loader)
    log_iter = int(iter_per_epoch / args.log_freq) + 1

    metrics: Dict[str, Tensor] = {"val_mIou": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "val_loss": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "train_mIou": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  "train_loss": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  }

    # ========== Training  =================
    for epoch in tqdm(range(args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_mIou, train_loss = do_epoch(args=args,
                                          train_loader=train_loader,
                                          iter_per_epoch=iter_per_epoch,
                                          model=model,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          epoch=epoch,
                                          callback=callback,
                                          log_iter=log_iter)

        val_mIou, val_loss = validate_fn(args=args,
                                         val_loader=val_loader,
                                         model=model,
                                         use_callback=False,
                                         suffix=f'train_{epoch}')
        if args.distributed:
            dist.all_reduce(val_mIou), dist.all_reduce(val_loss)
            val_mIou /= world_size
            val_loss /= world_size

        if main_process(args):
            # Live plot if desired with visdom
            if callback is not None:
                callback.scalar('val_loss', epoch, val_loss, title='Validiation Loss')
                callback.scalar('mIoU_val', epoch, val_mIou, title='Val metrics')

            # Model selection
            if val_mIou.item() > max_val_mIoU:
                max_val_mIoU = val_mIou.item()
                os.makedirs(savedir, exist_ok=True)
                filename = os.path.join(savedir, f'best.pth')
                if args.save_models:
                    print('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, filename)
            print("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

            # Sort and save the metrics
            for k in metrics:
                metrics[k][epoch] = eval(k)

            for k, e in metrics.items():
                path = os.path.join(savedir, f"{k}.npy")
                np.save(path, e.cpu().numpy())

    if args.save_models and main_process(args):
        filename = os.path.join(savedir, 'final.pth')
        print(f'Saving checkpoint to: {filename}')
        torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, filename)

    cleanup()


def cross_entropy(logits: torch.tensor,
                  one_hot: torch.tensor,
                  targets: torch.tensor,
                  mean_reduce: bool = True,
                  ignore_index: int = 255) -> torch.tensor:

    """
    inputs:
        one_hot  : shape [batch_size, num_classes, h, w]
        logits : shape [batch_size, num_classes, h, w]
        targets : shape [batch_size, h, w]
    returns:
        loss: shape [batch_size] or [] depending on mean_reduce

    """
    assert logits.size() == one_hot.size()
    log_prb = F.log_softmax(logits, dim=1)
    non_pad_mask = targets.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask)
    if mean_reduce:
        return loss.mean()  # average later
    else:
        return loss


def compute_loss(args: argparse.Namespace,
                 model: DDP,
                 images: torch.tensor,
                 targets: torch.tensor,
                 num_classes: int) -> torch.tensor:
    """
    inputs:
        images  : shape [batch_size, C, h, w]
        logits : shape [batch_size, num_classes, h, w]
        targets : shape [batch_size, h, w]

    returns:
        loss: shape []
        logits: shape [batch_size]

    """
    batch, h, w = targets.size()
    one_hot_mask = torch.zeros(batch, num_classes, h, w).to(dist.get_rank())
    new_target = targets.clone().unsqueeze(1)
    new_target[new_target == 255] = 0

    one_hot_mask.scatter_(1, new_target, 1).long()
    if args.smoothing:
        eps = 0.1
        one_hot = one_hot_mask * (1 - eps) + (1 - one_hot_mask) * eps / (num_classes - 1)
    else:
        one_hot = one_hot_mask  # [batch_size, num_classes, h, w]

    if args.mixup:
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(images.size()[0]).to(dist.get_rank())
        one_hot_a = one_hot
        targets_a = targets

        one_hot_b = one_hot[rand_index]
        target_b = targets[rand_index]
        mixed_images = lam * images + (1 - lam) * images[rand_index]

        logits = model(mixed_images)
        loss = cross_entropy(logits, one_hot_a, targets_a) * lam  \
            + cross_entropy(logits, one_hot_b, target_b) * (1. - lam)
    else:
        logits = model(images)
        loss = cross_entropy(logits, one_hot, targets)
    return loss


def do_epoch(args: argparse.Namespace,
             train_loader: torch.utils.data.DataLoader,
             model: DDP,
             optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler,
             epoch: int,
             callback: VisdomLogger,
             iter_per_epoch: int,
             log_iter: int) -> Tuple[torch.tensor, torch.tensor]:
    loss_meter = AverageMeter()
    train_losses = torch.zeros(log_iter).to(dist.get_rank())
    train_mIous = torch.zeros(log_iter).to(dist.get_rank())

    iterable_train_loader = iter(train_loader)

    if main_process(args):
        bar = tqdm(range(iter_per_epoch))
    else:
        bar = range(iter_per_epoch)

    for i in bar:
        model.train()
        current_iter = epoch * len(train_loader) + i + 1

        images, gt = iterable_train_loader.next()
        images = images.to(dist.get_rank(), non_blocking=True)
        gt = gt.to(dist.get_rank(), non_blocking=True)

        loss = compute_loss(args=args,
                            model=model,
                            images=images,
                            targets=gt.long(),
                            num_classes=args.num_classes_tr,
                            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.scheduler == 'cosine':
            scheduler.step()

        if i % args.log_freq == 0:
            model.eval()
            logits = model(images)
            intersection, union, target = intersectionAndUnionGPU(logits.argmax(1),
                                                                  gt,
                                                                  args.num_classes_tr,
                                                                  255)
            if args.distributed:
                dist.all_reduce(loss)
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            allAcc = (intersection.sum() / (target.sum() + 1e-10))  # scalar
            mAcc = (intersection / (target + 1e-10)).mean()
            mIoU = (intersection / (union + 1e-10)).mean()
            loss_meter.update(loss.item() / dist.get_world_size())

            if main_process(args):
                if callback is not None:
                    t = current_iter / len(train_loader)
                    callback.scalar('loss_train_batch', t, loss_meter.avg, title='Loss')
                    callback.scalars(['mIoU', 'mAcc', 'allAcc'], t,
                                     [mIoU, mAcc, allAcc],
                                     title='Training metrics')
                    for index, param_group in enumerate(optimizer.param_groups):
                        lr = param_group['lr']
                        callback.scalar('lr', t, lr, title='Learning rate')
                        break

                train_losses[int(i / args.log_freq)] = loss_meter.avg
                train_mIous[int(i / args.log_freq)] = mIoU

    if args.scheduler != 'cosine':
        scheduler.step()

    return train_mIous, train_losses


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        # args.epochs = 2
        args.n_runs = 2
        args.save_models = False

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)