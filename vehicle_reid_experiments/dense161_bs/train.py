import sys
import os
import argparse
import numpy as np
import random
import shutil
import time
import warnings
import pickle

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from thop import profile
from thop import clever_format
from tqdm import tqdm
from config import Config
from public.reid.densenet import denset161backbone
from public.reid.utils import get_logger, AverageMeter, accuracy
from public.reid.loss import TripletLoss, CenterLoss
from public.reid.val import get_dist_feature, generate_result_txt, Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--network_name',
                        type=str,
                        default=Config.network_name,
                        help='network name')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='num of batch size')
    parser.add_argument('--warm_up_epochs',
                        type=int,
                        default=Config.warm_up_epochs,
                        help='num of warm up epochs')
    parser.add_argument('--milestones',
                        type=list,
                        default=Config.milestones,
                        help='optimizer milestones')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoints,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger,
          args):
    top1 = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    iters = len(train_loader)

    for iter_index, sample in enumerate(train_loader):
        inputs, targets = sample["image"].cuda().float(
        ), sample["vehicle_id"].cuda()
        feat, feat_bn, feat_bn_fc = model(inputs)
        id_loss = criterion["id_loss"](feat_bn_fc, targets)
        triplet_loss, _, _ = criterion["triplet_loss"](feat, targets)
        center_loss = 0.0005 * (criterion["center_loss"](feat_bn, targets))
        loss = id_loss + triplet_loss + center_loss
        loss = loss / args.accumulation_steps

        loss.backward()

        if iter_index % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, _ = accuracy(feat_bn_fc, targets, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, top1 acc: {acc1.item():.2f}%, id_loss:{id_loss.item():.2f}, triplet_loss:{triplet_loss.item():.2f}, center_loss:{center_loss.item():.2f}, loss_total: {loss.item():.2f}"
            )

        iter_index += 1

    scheduler.step()

    return top1.avg, losses.avg


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    logger.info(f"creating model {args.network_name}")
    model = denset161backbone(args.pretrained, args.num_classes)

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    logger.info(
        f"model: '{args.network_name}', flops: {flops}, params: {params}")

    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")

    conv_and_fc_param_list, bn_param_list = [], []
    conv_and_fc_param_name_list, bn_param_name_list = [], []
    for name, param in model.named_parameters():
        if 'norm' in name:
            bn_param_list.append(param)
            bn_param_name_list.append(name)
        else:
            conv_and_fc_param_list.append(param)
            conv_and_fc_param_name_list.append(name)
    logger.info(f"{conv_and_fc_param_name_list}")
    logger.info(f"{bn_param_name_list}")

    weight_decay_setting_list = [
        {
            'params': conv_and_fc_param_list,
            'weight_decay': args.weight_decay,
        },
        {
            'params': bn_param_list,
            'weight_decay': 0.0,
        },
    ]

    model = model.cuda()
    optimizer = torch.optim.Adam(weight_decay_setting_list, lr=args.lr)
    # warm_up_with_multistep_lr
    warm_up_with_multistep_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.1**len(
        [m for m in args.milestones if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warm_up_with_multistep_lr)

    criterion_list = {
        "id_loss": nn.CrossEntropyLoss().cuda(),
        "triplet_loss": TripletLoss(margin=0.3),
        "center_loss": CenterLoss(num_classes=args.num_classes).cuda()
    }

    model = nn.DataParallel(model)

    best_map, best_top1 = 0.0, 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map, best_top1 = checkpoint['best_map'], checkpoint['best_top1']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc1']}%,"
            f"best_map: {checkpoint['best_map']},best_top1:{checkpoint['best_top1']}"
        )

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        acc1, losses = train(Config.train_loader, model, criterion_list,
                             optimizer, scheduler, epoch, logger, args)
        logger.info(
            f"train: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, losses: {losses:.2f}"
        )

        if epoch >= 60 and epoch % 2 == 0:
            dists, query_features, gallery_features = get_dist_feature(
                Config.val_loader, model, epoch, norm_feature=True)
            txt = generate_result_txt(dists,
                                      query_features,
                                      gallery_features,
                                      Config.val_dataset_pkl,
                                      color_bias=400,
                                      type_bias=400,
                                      group_threhold=0.05,
                                      group_rerank=True)
            evl = Evaluator(Config.val_dataset_pkl)
            effi = evl.eval_from_txt(txt)
            logger.info(
                f"epoch:{epoch},mAP:{effi['mAP']},CMC_1:{effi['CMC_1']}")
            if effi['mAP'] > best_map and effi['CMC_1'] > best_top1:
                logger.info(
                    f"best model update,epoch:{epoch},mAP:{effi['mAP']},CMC_1:{effi['CMC_1']}"
                )
                best_map, best_top1 = effi['mAP'], effi['CMC_1']
                logger.info("update best_map and best_top1")
                pickle.dump(
                    dists,
                    open(
                        os.path.join(
                            args.checkpoints,
                            '{}_best_dist.pkl'.format(args.network_name)),
                        "wb"))
                pickle.dump(
                    query_features,
                    open(
                        os.path.join(
                            args.checkpoints,
                            '{}_best_query_f.pkl'.format(args.network_name)),
                        "wb"))
                pickle.dump(
                    gallery_features,
                    open(
                        os.path.join(
                            args.checkpoints,
                            '{}_best_gallery_f.pkl'.format(args.network_name)),
                        "wb"))
                with open(
                        os.path.join(
                            args.checkpoints,
                            '{}_best_result.txt'.format(args.network_name)),
                        'w') as f:
                    f.write(txt)

                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.checkpoints, "{}_best_model.pth".format(
                        args.network_name)))

        # remember best prec@1 and save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'acc1': acc1,
                'loss': losses,
                'best_map': best_map,
                'best_top1': best_top1,
                'lr': scheduler.get_lr()[0],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.checkpoints, 'latest.pth'))

    logger.info(
        f"finish training, best_map: {best_map:.4f}, best_top1:{best_top1:.4f}"
    )
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(__name__, args.log)
    main(logger, args)