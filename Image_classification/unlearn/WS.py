import sys
import time

import torch
import utils

import itertools
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


@iterative_unlearn
def WS(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    #print(len(forget_loader))
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for (i, (image_f, target_f)), (j, (image_r, target_r)) in zip(enumerate(forget_loader), enumerate(retain_loader)):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            image_f = image_f.cuda()
            target_f = target_f.cuda()

            # compute output
            output_clean_f = model(image_f)
            loss_f = -criterion(output_clean_f, target_f)

            image_r = image_r.cuda()
            target_r = target_r.cuda()
            
            # compute output
            output_clean_r = model(image_r)
            loss_r = criterion(output_clean_r, target_r)

            loss = args.dcgd_alpha*loss_f + loss_r
            #print(f"loss_f: {loss_f}, loss_r: {loss_r}, loss: {loss}")
            #loss = loss_r
            optimizer.zero_grad()
            loss.backward()
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            

            optimizer.step()


            output = output_clean_f.float()
            loss = loss_f.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target_f)[0]

            losses.update(loss.item(), image_f.size(0))
            top1.update(prec1.item(), image_f.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()            

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg

````