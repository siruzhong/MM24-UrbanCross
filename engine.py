import time
import torch
import numpy as np
import itertools
from torch.autograd import Variable
import utils.utils as utils
import os
import shutil

# import tensorboard_logger as tb_logger
import wandb

# import logging
from loguru import logger
from torch.nn.utils.clip_grad import clip_grad_norm
from tqdm import tqdm


def train(args, train_loader, model, optimizer, epoch):
    """
    Train function to train the model using the provided training data.

    Args:
        args (argparse.Namespace): Parsed arguments.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    # Extract values from args
    grad_clip = args.grad_clip
    max_violation = args.max_violation
    margin = args.margin
    print_freq = args.print_freq
    if args.distributed:
        mean_loss = torch.zeros(1).to(args.gpuid)

    # Set model to train mode
    model.train()

    # Initialize average meters for tracking batch and data loading time
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()

    # Initialize log collector
    train_logger = utils.LogCollector()

    # Initialize timer
    end = time.time()

    # Get model parameters
    params = list(model.parameters())

    # Iterate over batches in the training data
    for i, train_data in enumerate(train_loader):
        # Unpack training data
        input_visual, ids, input_text, segment_imgs = train_data

        margin = float(margin)
        # Measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            input_visual = input_visual.cuda(args.gpuid)
            input_text = input_text.cuda(args.gpuid)
            segment_imgs = segment_imgs.cuda(args.gpuid)

        # Synchronize CUDA streams
        torch.cuda.synchronize(device=args.gpuid)

        if not args.il_measure:
            scores_img2text, scores_seg2text = model(
                input_visual, input_text, segment_imgs
            )
            loss_img2text = utils.calcul_contraloss(
                args,
                scores_img2text,
                input_visual.size(0),
                margin,
                max_violation=max_violation,
            )
            loss_seg2text = utils.calcul_contraloss(
                args,
                scores_seg2text,
                input_visual.size(0),
                margin,
                max_violation=max_violation,
            )
            loss = loss_img2text + loss_seg2text
        else:
            scores, scores_intra_img, scores_intra_cap = model(
                input_visual, input_text, lengths
            )
            intra_loss = utils.calcul_intraloss(
                args, scores_intra_img
            ) + utils.calcul_intraloss(args, scores_intra_cap)
            loss = (
                utils.calcul_contraloss(
                    args,
                    scores,
                    input_visual.size(0),
                    margin,
                    max_violation=max_violation,
                )
                + intra_loss
            )

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        # Log loss values
        wandb.log(
            {
                "loss": loss.cpu().data.numpy(),
                "loss_img2text": loss_img2text.cpu().data.numpy(),
                "loss_seg2text": loss_seg2text.cpu().data.numpy(),
            }
        )

        optimizer.zero_grad()
        loss.backward()
        if args.distributed:
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses
            train_logger.update("Loss", round(mean_loss.item(), 3))
        else:
            train_logger.update("Loss", loss.cpu().data.numpy())
            train_logger.update("Loss_img2text", loss_img2text.cpu().data.numpy())
            train_logger.update("Loss_seg2text", loss_seg2text.cpu().data.numpy())

        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # Update average meters
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training progress
        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                "Epoch [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}\t"
                "{elog}\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    elog=str(train_logger),
                )
            )
            utils.get_GPU_usage()

        wandb.log(
            {
                "epoch": epoch,
                "batch_time": batch_time.val,
            }
        )

        # Log loss values to W&B
        train_logger.wandb_log()


def train_without_sam(args, train_loader, model, optimizer, epoch):
    """
    Train function to train the model using the provided training data.

    Args:
        args (argparse.Namespace): Parsed arguments.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    # Extract values from args
    grad_clip = args.grad_clip
    max_violation = args.max_violation
    margin = args.margin
    print_freq = args.print_freq
    if args.distributed:
        mean_loss = torch.zeros(1).to(args.gpuid)

    # Set model to train mode
    model.train()

    # Initialize average meters for tracking batch and data loading time
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()

    # Initialize log collector
    train_logger = utils.LogCollector()

    # Initialize timer
    end = time.time()

    # Get model parameters
    params = list(model.parameters())

    # Iterate over batches in the training data
    for i, train_data in enumerate(train_loader):
        # Unpack training data
        input_visual, ids, input_text = train_data

        margin = float(margin)
        # Measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            input_visual = input_visual.cuda(args.gpuid)
            input_text = input_text.cuda(args.gpuid)

        # Synchronize CUDA streams
        torch.cuda.synchronize(device=args.gpuid)

        if not args.il_measure:
            scores_img2text = model(input_visual, input_text)
            loss_img2text = utils.calcul_contraloss(
                args,
                scores_img2text,
                input_visual.size(0),
                margin,
                max_violation=max_violation,
            )
            loss = loss_img2text
        else:
            scores, scores_intra_img, scores_intra_cap = model(
                input_visual, input_text, lengths
            )
            intra_loss = utils.calcul_intraloss(
                args, scores_intra_img
            ) + utils.calcul_intraloss(args, scores_intra_cap)
            loss = (
                utils.calcul_contraloss(
                    args,
                    scores,
                    input_visual.size(0),
                    margin,
                    max_violation=max_violation,
                )
                + intra_loss
            )

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        # Log loss values
        wandb.log(
            {
                "loss": loss.cpu().data.numpy(),
                "loss_img2text": loss_img2text.cpu().data.numpy(),
            }
        )

        optimizer.zero_grad()
        loss.backward()
        if args.distributed:
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses
            train_logger.update("Loss", round(mean_loss.item(), 3))
        else:
            train_logger.update("Loss", loss.cpu().data.numpy())
            train_logger.update("Loss_img2text", loss_img2text.cpu().data.numpy())

        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # Update average meters
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training progress
        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                "Epoch [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}\t"
                "{elog}\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    elog=str(train_logger),
                )
            )
            utils.get_GPU_usage()

        wandb.log(
            {
                "epoch": epoch,
                "batch_time": batch_time.val,
            }
        )

        # Log loss values to W&B
        train_logger.wandb_log()


def train_finetune(args, train_loader_source, train_loader_target, model, optimizer, epoch):
    # Extract values from arguments
    grad_clip = args.grad_clip
    max_violation = args.max_violation
    margin = args.margin
    print_freq = args.print_freq

    if args.distributed:
        mean_loss = torch.zeros(1).to(args.gpuid)

    # Switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())

    # Create an iterator for the target loader
    target_loader_cycle = itertools.cycle(train_loader_target)
    num_cycle_of_target = -1

    for i, source_data in enumerate(train_loader_source):
        images_source, cap_tokens_source = source_data
        images_target, cap_tokens_target = next(target_loader_cycle)

        if i % len(train_loader_target) == 0:
            num_cycle_of_target += 1

        batch_size = images_source.size(0)
        margin = float(margin)

        # Measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visuals_source = images_source
        input_visuals_target = images_target
        input_text_source = cap_tokens_source
        input_text_target = cap_tokens_target

        if torch.cuda.is_available():
            input_visuals_source = input_visuals_source.cuda(args.gpuid)
            input_visuals_target = input_visuals_target.cuda(args.gpuid)
            input_text_source = input_text_source.cuda(args.gpuid)
            input_text_target = input_text_target.cuda(args.gpuid)

        torch.cuda.synchronize(device=args.gpuid)

        # Calculate clip_loss, adv_loss, and filter_ratio
        clip_loss, adv_loss, filter_ratio = model(
            input_visuals_source,
            input_visuals_target,
            input_text_source,
            input_text_target,
            num_cycle_of_target=num_cycle_of_target,
        )
        loss = clip_loss + adv_loss

        # Clip gradients if grad_clip is positive
        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        # Log the loss to Weights and Biases
        wandb.log(
            {
                "loss": loss.cpu().data.numpy(),
                "loss_clip": clip_loss.cpu().data.numpy(),
                "loss_adv": adv_loss.cpu().data.numpy(),
            }
        )

        # Zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()

        if args.distributed:
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses
            train_logger.update("Loss", round(mean_loss.item(), 3))
        else:
            if args.il_measure:
                train_logger.update("Loss_avg", loss.cpu().data.numpy())

        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                "Epoch [{0}][{1}/{2}(source)][{1}/{3}(target)]\t"
                "Time {batch_time.val:.3f}\t"
                "{elog}\t".format(
                    epoch,
                    i,
                    len(train_loader_source),
                    len(train_loader_target),
                    batch_time=batch_time,
                    elog=str(train_logger),
                )
            )
            logger.info(f"{num_cycle_of_target}_th cycle of target data")
            logger.info(f"filter_ratio: {filter_ratio}")
            utils.get_GPU_usage()

        # Log the batch time to Weights and Biases
        wandb.log(
            {
                "batch_time": batch_time.val,
            }
        )
        train_logger.wandb_log()


def train_finetune_curriculum(
    args, train_loader_source, train_loader_target, model, optimizer, epoch
):
    grad_clip = args.grad_clip
    max_violation = args.max_violation
    margin = args.margin
    print_freq = args.print_freq
    if args.distributed:
        mean_loss = torch.zeros(1).to(args.gpuid)

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())

    # 创建 B_loader 的循环迭代器
    target_loader_cycle = itertools.cycle(train_loader_target)
    source_loader_cycle = itertools.cycle(train_loader_source)

    num_cycle_of_target = 0
    i = 0
    while num_cycle_of_target <= 4:
        images_source, cap_tokens_source = next(source_loader_cycle)
        images_target, cap_tokens_target = next(target_loader_cycle)
        if i % len(train_loader_target) == 0 and i != 0:
            num_cycle_of_target += 1
        if num_cycle_of_target > 4:
            break

        i += 1

        batch_size = images_source.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visuals_source = images_source
        input_visuals_target = images_target
        input_text_source = cap_tokens_source
        input_text_target = cap_tokens_target

        if torch.cuda.is_available():
            input_visuals_source = input_visuals_source.cuda(args.gpuid)
            input_visuals_target = input_visuals_target.cuda(args.gpuid)
            input_text_source = input_text_source.cuda(args.gpuid)
            input_text_target = input_text_target.cuda(args.gpuid)

        torch.cuda.synchronize(device=args.gpuid)

        clip_loss, adv_loss, filter_ratio = model(
            input_visuals_source,
            input_visuals_target,
            input_text_source,
            input_text_target,
            num_cycle_of_target=num_cycle_of_target,
        )
        loss = clip_loss + adv_loss

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        wandb.log({"loss": loss.cpu().data.numpy()})
        optimizer.zero_grad()
        loss.backward()
        if args.distributed:
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses

            train_logger.update("Loss", round(mean_loss.item(), 3))
        else:
            train_logger.update("Loss_avg", loss.cpu().data.numpy())

        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                "Epoch [{0}][{1}/{2}(source)][{1}/{3}(target)]\t"
                "Time {batch_time.val:.3f}\t"
                "{elog}\t".format(
                    epoch,
                    i,
                    len(train_loader_source),
                    len(train_loader_target),
                    batch_time=batch_time,
                    elog=str(train_logger),
                )
            )
            logger.info(f"{num_cycle_of_target}_th cycle of target data")
            logger.info(f"filter_ratio: {filter_ratio}")
            utils.get_GPU_usage()

        wandb.log(
            {
                "epoch": epoch,
                "batch_time": batch_time.val,
            }
        )
        train_logger.wandb_log()


def validate(args, val_loader, model):
    print("")
    print(
        "--------------------- Start validation on training set ---------------------"
    )
    model.eval()  # Set the model to evaluation mode

    val_logger = utils.LogCollector()  # Create a logger to collect logs
    model.logger = val_logger  # Set the logger for the model

    start = time.time()  # Record start time

    # Prepare variables to store input data
    input_visual = []  # For image data
    input_text = []  # For text data
    input_seg = []  # For segmentation data

    # Iterate through the validation data loader to get data
    for idx, val_data in enumerate(itertools.islice(val_loader, 3)):
        images, ids, cap_tokens, segment_img = val_data  # Unpack data
        input_visual.append(images)  # Store image data
        input_text.append(cap_tokens)  # Store text data
        input_seg.append(segment_img)  # Store segmentation data

    # Convert data to tensor and concatenate
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)
    input_seg = torch.cat(input_seg, dim=0)

    # Perform inference using the model
    d = utils.shard_dis_mine(args, input_visual, input_text, input_seg, model)
    end = time.time()  # Record end time

    print(
        "Calculate similarity time: {:.4f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.4f} r5i:{:.4f} r10i:{:.4f} medri:{:.4f} meanri:{:.4f}\n"
        "t2i => r1t:{:.4f} r5t:{:.4f} r10t:{:.4f} medrt:{:.4f} meanrt:{:.4f}\n"
        "mR:{:.4f}".format(
            r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
        )
    )

    print("--------------------- End validation on training set ---------------------")
    print("")

    # Log evaluation results
    wandb.log(
        {
            "val/r1i": r1i,
            "val/r5i": r5i,
            "val/r10i": r10i,
            "val/medri": medri,
            "val/meanri": meanri,
            "val/r1t": r1t,
            "val/r5t": r5t,
            "val/r10t": r10t,
            "val/medrt": medrt,
            "val/meanrt": meanrt,
            "val/rsum": currscore,
        }
    )

    return currscore, all_score


def validate_without_sam(args, val_loader, model):
    print("")
    print(
        "--------------------- Start validation on training set without sam---------------------"
    )
    model.eval()  # Set the model to evaluation mode

    val_logger = utils.LogCollector()  # Create a logger to collect logs
    model.logger = val_logger  # Set the logger for the model

    start = time.time()  # Record start time

    # Prepare variables to store input data
    input_visual = []  # For image data
    input_text = []  # For text data

    # Iterate through the validation data loader to get data
    for idx, val_data in enumerate(itertools.islice(val_loader, 3)):
        images, ids, cap_tokens = val_data  # Unpack data
        input_visual.append(images)  # Store image data
        input_text.append(cap_tokens)  # Store text data

    # Convert data to tensor and concatenate
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)

    # Perform inference using the model
    d = utils.shard_dis_without_sam_mine(args, input_visual, input_text, model)
    end = time.time()  # Record end time

    print(
        "Calculate similarity time: {:.4f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.4f} r5i:{:.4f} r10i:{:.4f} medri:{:.4f} meanri:{:.4f}\n"
        "t2i => r1t:{:.4f} r5t:{:.4f} r10t:{:.4f} medrt:{:.4f} meanrt:{:.4f}\n"
        "mR:{:.4f}".format(
            r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
        )
    )

    print("--------------------- End validation on training set ---------------------")
    print("")

    # Log evaluation results
    wandb.log(
        {
            "val/r1i": r1i,
            "val/r5i": r5i,
            "val/r10i": r10i,
            "val/medri": medri,
            "val/meanri": meanri,
            "val/r1t": r1t,
            "val/r5t": r5t,
            "val/r10t": r10t,
            "val/medrt": medrt,
            "val/meanrt": meanrt,
            "val/rsum": currscore,
        }
    )

    return currscore, all_score


def validate_finetune(args, val_loader, model):
    logger.info("--------------------- start val on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()

    input_visual = []
    input_text = []
    for idx, val_data in enumerate(val_loader):
        images, cap_tokens = val_data
        input_visual.append(images)
        input_text.append(cap_tokens)
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)

    d = utils.shard_dis_mine_finetune(
        args,
        input_visual,
        input_text,
        model,
    )
    end = time.time()
    print("calculate similarity time: {:.4f} s".format(end - start))

    # image to text
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    # text to image
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # import ipdb; ipdb.set_trace()
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = (
        "i2t => r1i:{:.4f} r5i:{:.4f} r10i:{:.4f} medri:{:.4f} meanri:{:.4f}\n"
        "t2i => r1t:{:.4f} r5t:{:.4f} r10t:{:.4f} medrt:{:.4f} meanrt:{:.4f}\n"
        "mR:{:.4f}".format(
            r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
        )
    )

    logger.info("--------------------- end val on training ---------------------")
    wandb.log(
        {
            "val/r1i": r1i,
            "val/r5i": r5i,
            "val/r10i": r10i,
            "val/medri": medri,
            "val/meanri": meanri,
            "val/r1t": r1t,
            "val/r5t": r5t,
            "val/r10t": r10t,
            "val/medrt": medrt,
            "val/meanrt": meanrt,
            "val/rsum": currscore,
        }
    )

    return currscore, all_score


def validate_test(args, test_loader, model):
    print("")
    print("--------------------- Start testing on training set ---------------------")
    model.eval()  # Set the model to evaluation mode

    val_logger = utils.LogCollector()  # Create a logger to collect logs
    model.logger = val_logger  # Set the logger for the model

    start = time.time()  # Record start time

    input_visual = []  # For storing image data
    input_text = []  # For storing text data
    input_seg = []  # For storing segmentation data

    # Iterate through the test data loader to get data
    # for idx, val_data in enumerate(tqdm(itertools.islice(test_loader, 3))):
    for idx, val_data in enumerate(tqdm(test_loader)):
        images, ids, cap_tokens, segment_img = val_data  # Unpack data
        input_visual.append(images)  # Store image data
        input_text.append(cap_tokens)  # Store text data
        input_seg.append(segment_img)  # Store segmentation data

    # Convert data to tensor and concatenate
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)
    input_seg = torch.cat(input_seg, dim=0)

    # Perform inference using the model
    d = utils.shard_dis_mine(args, input_visual, input_text, input_seg, model)
    end = time.time()  # Record end time

    print(
        "Calculate similarity time: {:.4f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.4f} r5i:{:.4f} r10i:{:.4f} medri:{:.4f} meanri:{:.4f}\n"
        "t2i => r1t:{:.4f} r5t:{:.4f} r10t:{:.4f} medrt:{:.4f} meanrt:{:.4f}\n"
        "mR:{:.4f}".format(
            r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
        )
    )

    print("--------------------- End testing on training set ---------------------")
    print("")

    # Log evaluation results
    wandb.log(
        {
            "test/r1i": r1i,
            "test/r5i": r5i,
            "test/r10i": r10i,
            "test/medri": medri,
            "test/meanri": meanri,
            "test/r1t": r1t,
            "test/r5t": r5t,
            "test/r10t": r10t,
            "test/medrt": medrt,
            "test/meanrt": meanrt,
            "test/rsum": currscore,
        }
    )

    return currscore, all_score


def validate_test_without_sam(args, test_loader, model):
    print("")
    print("--------------------- Start testing on training set ---------------------")
    model.eval()  # Set the model to evaluation mode

    val_logger = utils.LogCollector()  # Create a logger to collect logs
    model.logger = val_logger  # Set the logger for the model

    start = time.time()  # Record start time

    input_visual = []  # For storing image data
    input_text = []  # For storing text data

    # Iterate through the test data loader to get data
    # for idx, val_data in enumerate(tqdm(itertools.islice(test_loader, 3))):
    for idx, val_data in enumerate(tqdm(test_loader)):
        images, ids, cap_tokens = val_data  # Unpack data
        input_visual.append(images)  # Store image data
        input_text.append(cap_tokens)  # Store text data

    # Convert data to tensor and concatenate
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)

    # Perform inference using the model
    d = utils.shard_dis_without_sam_mine(args, input_visual, input_text, model)
    end = time.time()  # Record end time

    print(
        "Calculate similarity time: {:.4f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.4f} r5i:{:.4f} r10i:{:.4f} medri:{:.4f} meanri:{:.4f}\n"
        "t2i => r1t:{:.4f} r5t:{:.4f} r10t:{:.4f} medrt:{:.4f} meanrt:{:.4f}\n"
        "mR:{:.4f}".format(
            r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
        )
    )

    print("--------------------- End testing on training set ---------------------")
    print("")

    # Log evaluation results
    wandb.log(
        {
            "test/r1i": r1i,
            "test/r5i": r5i,
            "test/r10i": r10i,
            "test/medri": medri,
            "test/meanri": meanri,
            "test/r1t": r1t,
            "test/r5t": r5t,
            "test/r10t": r10t,
            "test/medrt": medrt,
            "test/meanrt": meanrt,
            "test/rsum": currscore,
        }
    )

    return currscore, all_score


def test(args, test_loader, model):
    print("")
    print("--------------------- start test ---------------------")
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))

    input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    input_text_length = [0] * len(test_loader.dataset)

    embed_start = time.time()
    for i, val_data in enumerate(test_loader):

        images, captions, lengths, ids = val_data

        for id, img, cap, l in zip(
            ids, (images.numpy().copy()), (captions.numpy().copy()), lengths
        ):
            input_visual[id] = img

            input_text[id, : captions.size(1)] = cap
            input_text_length[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    embed_end = time.time()
    print("## embedding time: {:.4f} s".format(embed_end - embed_start))

    d = utils.shard_dis_SWAN(
        args, input_visual, input_text, model, lengths=input_text_length
    )

    end = time.time()
    print("calculate similarity time: {:.4f} s".format(end - start))
    print("--------------------- end test ---------------------")
    print("")
    return d


def test_mine(args, test_loader, model):
    print("")
    print("--------------------- start test on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()

    input_visual = []
    input_text = []
    img_paths = []
    captions = []

    # for idx, val_data in enumerate(tqdm(itertools.islice(test_loader, 20))):
    for idx, val_data in enumerate(tqdm(test_loader)):
        images, cap_tokens, img_path, caption = val_data
        input_visual.append(images)
        input_text.append(cap_tokens)
        img_paths.extend(img_path)
        captions.extend(caption)

    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)

    logger.info("begin to compute distance")
    d = utils.shard_dis_mine_finetune(args, input_visual, input_text, model)
    
    # # Top 10 visualization
    # # Normalize the distance values to be between 0 and 1
    # d_normalized = (d - np.min(d)) / (np.max(d) - np.min(d))
    
    # # Iterate through each image path
    # for i in tqdm(range(len(img_paths))):
    #     # Find the indices of the top 10 captions with the highest distance values
    #     top10_indices = np.argsort(-d_normalized[i])[:10]
        
    #     # Get the top 10 captions and their corresponding distance values
    #     top10_captions = [captions[idx] for idx in top10_indices]
    #     top10_values = [d_normalized[i][idx] for idx in top10_indices]

    #     # Create a save path for the current image and ensure the directory exists
    #     savepath = os.path.join(args.ckpt_save_path, img_paths[i].split("/")[-1])
    #     os.makedirs(savepath, exist_ok=True)
        
    #     # Copy the current image to the save path
    #     shutil.copy(img_paths[i], savepath)
        
    #     # Write the top 10 captions and their corresponding distance values to a text file
    #     with open(os.path.join(savepath, "top10_captions.txt"), "w") as f:
    #         for j in range(10):
    #             f.write(f"{top10_captions[j]}\n")
    #             f.write(f"{top10_values[j]}\n")
    #             f.write("\n")
    
    end = time.time()
    print("calculate similarity time: {:.4f} s".format(end - start))

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = (
        "i2t => r1i:{:.4f} r5i:{:.4f} r10i:{:.4f} medri:{:.4f} meanri:{:.4f}\n"
        "t2i => r1t:{:.4f} r5t:{:.4f} r10t:{:.4f} medrt:{:.4f} meanrt:{:.4f}\n"
        "mR:{:.4f}".format(
            r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
        )
    )

    print("--------------------- end test on training ---------------------")
    print("")

    wandb.log(
        {
            "test/r1i": r1i,
            "test/r5i": r5i,
            "test/r10i": r10i,
            "test/medri": medri,
            "test/meanri": meanri,
            "test/r1t": r1t,
            "test/r5t": r5t,
            "test/r10t": r10t,
            "test/medrt": medrt,
            "test/meanrt": meanrt,
            "test/rsum": currscore,
        }
    )
    return currscore, all_score


def save(args, test_loader, model):
    print("")
    print("--------------------- start test ---------------------")
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))

    input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    input_text_length = [0] * len(test_loader.dataset)

    for i, val_data in enumerate(test_loader):

        images, captions, lengths, ids = val_data

        for id, img, cap, l in zip(
            ids, (images.numpy().copy()), (captions.numpy().copy()), lengths
        ):
            input_visual[id] = img

            input_text[id, : captions.size(1)] = cap
            input_text_length[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    img_emb, text_emb = utils.save_img_text_emb(
        args, input_visual, input_text, model, lengths=input_text_length
    )

    return img_emb, text_emb
