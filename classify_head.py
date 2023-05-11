import argparse
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.models.layers import trunc_normal_
from utils.lars import LARS

import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed

import models.models_vit as models_vit

from utils import lr_sched
from dataset.imbalance_cifar import ImbalanceCIFAR10
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_cifar', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./log/classify_head',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./classify_head',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./output_dir/checkpoint-99.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    tf_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "log"))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = ImbalanceCIFAR10(
            root=args.data_path, train=True, download=True, transform=transform_train
        )
    
    num_per_class = torch.tensor(train_dataset.get_cls_num_list())
    val_dataset = datasets.CIFAR10(root=args.data_path,
                                       train=False, download=True, transform=transform_val)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.num_workers, 
                                               pin_memory=args.pin_mem)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)

    
    # define the model
    model = models_vit.__dict__[args.model](num_classes=10)

    if args.start_epoch == 0:
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # for linear prob only
        # hack: revise model's head with BN
        trunc_normal_(model.head.weight, std=0.01)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    else:
        model = torch.load(f"./classify_output_dir/epoch-{args.start_epoch}")
    
    
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    model.to(device)

    args.lr = args.blr * args.batch_size / 256

    optimizer = LARS(model.head.parameters(), lr=args.lr, weight_decay=0)


    beta = 0.9999
    effective_num = 1.0 - np.power(beta, num_per_class)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_per_class)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    losser = torch.nn.CrossEntropyLoss(per_cls_weights)

    

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        model.train()
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            
            lr_sched.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
            # adjust_learning_rate(optimizer, epoch, args)
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = losser(pred, label)
            
            acc1, acc3 = accuracy(pred, label, topk=(1, 3))
            losses.update(loss.item(), img.size(0))
            top1.update(acc1[0], img.size(0))
            top3.update(acc3[0], img.size(0))         

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1, top3=top3,
                    lr=optimizer.param_groups[-1]['lr']))
                print(output)
            
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top3', top3.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
        
            
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            validate(val_loader, model, losser, epoch, tf_writer)
            torch.save(model, f"./classify_head/epoch-{epoch}")



def validate(val_loader, model, criterion, epoch, tf_writer, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (img, label) in enumerate(val_loader):
            img = img.cuda()
            label = label.cuda()

            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = criterion(pred, label)

            acc1, acc3 = accuracy(pred, label, topk=(1, 3))
            losses.update(loss.item(), img.size(0))
            top1.update(acc1[0], img.size(0))
            top3.update(acc3[0], img.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(pred, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(label.cpu().numpy())

            if i % 20 == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top3=top3))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top3=top3, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)


        if tf_writer is not None:
            tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top3', top3.avg, epoch)
            tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg 


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
