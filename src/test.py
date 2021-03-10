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
from collections import defaultdict
from .dataset.dataset import get_val_loader
from .util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, main_process
from .util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from .classifier import Classifier
from .model.pspnet import get_model
import torch.distributed as dist
from tqdm import tqdm
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from .visu import make_episode_visualization
from typing import Tuple


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Testing')
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

    print(f"==> Running DDP checkpoint example on rank {rank}.")
    setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed + rank)
        np.random.seed(args.manual_seed + rank)
        torch.manual_seed(args.manual_seed + rank)
        torch.cuda.manual_seed_all(args.manual_seed + rank)
        random.seed(args.manual_seed + rank)

    # ========== Model  ==========
    model = get_model(args).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    root = get_model_dir(args)

    if args.ckpt_used is not None:
        filepath = os.path.join(root, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Data  ==========
    episodic_val_loader, _ = get_val_loader(args)

    # ========== Test  ==========
    val_Iou, val_loss = episodic_validate(args=args,
                                          val_loader=episodic_val_loader,
                                          model=model,
                                          use_callback=(args.visdom_port != -1),
                                          suffix=f'test')
    if args.distributed:
        dist.all_reduce(val_Iou), dist.all_reduce(val_loss)
        val_Iou /= world_size
        val_loss /= world_size

    cleanup()


def episodic_validate(args: argparse.Namespace,
                      val_loader: torch.utils.data.DataLoader,
                      model: DDP,
                      use_callback: bool,
                      suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Start testing')

    model.eval()
    nb_episodes = int(args.test_num / args.batch_size_val)

    # ========== Metrics initialization  ==========

    H, W = args.image_size, args.image_size
    c = model.module.bottleneck_dim
    h = model.module.feature_res[0]
    w = model.module.feature_res[1]

    runtimes = torch.zeros(args.n_runs)
    deltas_init = torch.zeros((args.n_runs, nb_episodes, args.batch_size_val))
    deltas_final = torch.zeros((args.n_runs, nb_episodes, args.batch_size_val))
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ========== Perform the runs  ==========
    for run in tqdm(range(args.n_runs)):

        # =============== Initialize the metric dictionaries ===============

        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        IoU = defaultdict(int)

        # =============== episode = group of tasks ===============
        runtime = 0
        for e in tqdm(range(nb_episodes)):
            t0 = time.time()
            features_s = torch.zeros(args.batch_size_val, args.shot, c, h, w).to(dist.get_rank())
            features_q = torch.zeros(args.batch_size_val, 1, c, h, w).to(dist.get_rank())
            gt_s = 255 * torch.ones(args.batch_size_val, args.shot, args.image_size,
                                    args.image_size).long().to(dist.get_rank())
            gt_q = 255 * torch.ones(args.batch_size_val, 1, args.image_size,
                                    args.image_size).long().to(dist.get_rank())
            n_shots = torch.zeros(args.batch_size_val).to(dist.get_rank())
            classes = []  # All classes considered in the tasks

            # =========== Generate tasks and extract features for each task ===============
            with torch.no_grad():
                for i in range(args.batch_size_val):
                    try:
                        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iter_loader.next()
                    except:
                        iter_loader = iter(val_loader)
                        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iter_loader.next()
                    iter_num += 1

                    q_label = q_label.to(dist.get_rank(), non_blocking=True)
                    spprt_imgs = spprt_imgs.to(dist.get_rank(), non_blocking=True)
                    s_label = s_label.to(dist.get_rank(), non_blocking=True)
                    qry_img = qry_img.to(dist.get_rank(), non_blocking=True)

                    f_s = model.module.extract_features(spprt_imgs.squeeze(0))
                    f_q = model.module.extract_features(qry_img)

                    shot = f_s.size(0)
                    n_shots[i] = shot
                    features_s[i, :shot] = f_s.detach()
                    features_q[i] = f_q.detach()
                    gt_s[i, :shot] = s_label
                    gt_q[i, 0] = q_label
                    classes.append([class_.item() for class_ in subcls])

            # =========== Normalize features along channel dimension ===============
            if args.norm_feat:
                features_s = F.normalize(features_s, dim=2)
                features_q = F.normalize(features_q, dim=2)

            # =========== Create a callback is args.visdom_port != -1 ===============
            callback = VisdomLogger(port=args.visdom_port) if use_callback else None

            # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
            classifier = Classifier(args)
            classifier.init_prototypes(features_s, features_q, gt_s, gt_q, classes, callback)
            batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
            deltas_init[run, e, :] = batch_deltas.cpu()

            # =========== Perform RePRI inference ===============
            batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback)
            deltas_final[run, e, :] = batch_deltas
            t1 = time.time()
            runtime += t1 - t0
            logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w]
            logits = F.interpolate(logits,
                                   size=(H, W),
                                   mode='bilinear',
                                   align_corners=True)
            probas = classifier.get_probas(logits).detach()
            intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2)  # [n_tasks, shot, num_class]
            intersection, union = intersection.cpu(), union.cpu()

            # ================== Log metrics ==================
            one_hot_gt = to_one_hot(gt_q, 2)
            valid_pixels = gt_q != 255
            loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
            loss_meter.update(loss.item())
            for i, task_classes in enumerate(classes):
                for j, class_ in enumerate(task_classes):
                    cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                    cls_union[class_] += union[i, 0, j + 1]

            for class_ in cls_union:
                IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)

            if (iter_num % 200 == 0):
                mIoU = np.mean([IoU[i] for i in IoU])
                print('Test: [{}/{}] '
                      'mIoU {:.4f} '
                      'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
                                                                                 args.test_num,
                                                                                 mIoU,
                                                                                 loss_meter=loss_meter,
                                                                                 ))

            # ================== Visualization ==================
            if args.visu:
                root = os.path.join('plots', 'episodes')
                os.makedirs(root, exist_ok=True)
                save_path = os.path.join(root, f'run_{run}_episode_{e}.pdf')
                make_episode_visualization(img_s=spprt_imgs[0].cpu().numpy(),
                                           img_q=qry_img[0].cpu().numpy(),
                                           gt_s=s_label[0].cpu().numpy(),
                                           gt_q=q_label[0].cpu().numpy(),
                                           preds=probas[-1].cpu().numpy(),
                                           save_path=save_path)


        runtimes[run] = runtime
        mIoU = np.mean(list(IoU.values()))
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    # ================== Save metrics ==================
    if args.save_oracle:
        root = os.path.join('plots', 'oracle')
        os.makedirs(root, exist_ok=True)
        np.save(os.path.join(root, 'delta_init.npy'), deltas_init.numpy())
        np.save(os.path.join(root, 'delta_final.npy'), deltas_final.numpy())

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()


def standard_validate(args: argparse.Namespace,
                      val_loader: torch.utils.data.DataLoader,
                      model: DDP,
                      use_callback: bool,
                      suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Standard validation')
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    iterable_val_loader = iter(val_loader)

    bar = tqdm(range(len(iterable_val_loader)))

    loss = 0.
    intersections = torch.zeros(args.num_classes_tr).to(dist.get_rank())
    unions = torch.zeros(args.num_classes_tr).to(dist.get_rank())

    with torch.no_grad():
        for i in bar:
            images, gt = iterable_val_loader.next()
            images = images.to(dist.get_rank(), non_blocking=True)
            gt = gt.to(dist.get_rank(), non_blocking=True)
            logits = model(images).detach()
            loss += loss_fn(logits, gt)
            intersection, union, _ = intersectionAndUnionGPU(logits.argmax(1),
                                                             gt,
                                                             args.num_classes_tr,
                                                             255)
            intersections += intersection
            unions += union
        loss /= len(val_loader.dataset)

    if args.distributed:
        dist.all_reduce(loss)
        dist.all_reduce(intersections)
        dist.all_reduce(unions)

    mIoU = (intersections / (unions + 1e-10)).mean()
    loss /= dist.get_world_size()
    return mIoU, loss


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)