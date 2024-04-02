import time
import torch
import numpy as np
import itertools
from torch.autograd import Variable
import utils

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
            scores_img2text, scores_seg2text = model(input_visual, input_text, segment_imgs)
            loss_img2text = utils.calcul_contraloss(args, scores_img2text, input_visual.size(0), margin, max_violation=max_violation)
            loss_seg2text = utils.calcul_contraloss(args, scores_seg2text, input_visual.size(0), margin, max_violation=max_violation)
            loss = loss_img2text + loss_seg2text
        else:
            scores, scores_intra_img, scores_intra_cap = model(input_visual, input_text, lengths)
            intra_loss = utils.calcul_intraloss(args, scores_intra_img) + utils.calcul_intraloss(args, scores_intra_cap)
            loss = (utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation,) + intra_loss)
            
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
            loss_img2text = utils.calcul_contraloss(args, scores_img2text, input_visual.size(0), margin, max_violation=max_violation)
            loss = loss_img2text
        else:
            scores, scores_intra_img, scores_intra_cap = model(input_visual, input_text, lengths)
            intra_loss = utils.calcul_intraloss(args, scores_intra_img) + utils.calcul_intraloss(args, scores_intra_cap)
            loss = (utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation,) + intra_loss)
            
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


def train_finetune(
    args, train_loader_source, train_loader_target, model, optimizer, epoch
):

    # extract value
    grad_clip = args.grad_clip
    max_violation = args.max_violation
    margin = args.margin
    # loss_name = args.model_name + "_" + args.data_name
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
    iter_target = iter(train_loader_target)
    for i, train_data in enumerate(train_loader_source):
        # images, ids, cap_tokens, segment_imgs, tag_tokens = train_data
        # images, ids, cap_tokens, segment_imgs = train_data
        images_source, ids, cap_tokens_source = train_data
        images_target, ids, cap_tokens_target = next(iter_target)
        # import ipdb; ipdb.set_trace()
        # images, ids, cap_tokens, segment_img, tag_tokens

        batch_size = images_source.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visuals_source = images_source
        input_visuals_target = images_target
        # import ipdb; ipdb.set_trace()
        # segment_imgs = Variable(segment_imgs)
        # input_text = Variable(captions)
        input_text_source = cap_tokens_source
        input_text_target = cap_tokens_target
        # input_tags = Variable(tag_tokens)
        # import ipdb;ipdb.set_trace()

        if torch.cuda.is_available():
            input_visuals_source = input_visuals_source.cuda(args.gpuid)
            input_visuals_target = input_visuals_target.cuda(args.gpuid)
            input_text_source = input_text_source.cuda(args.gpuid)
            input_text_target = input_text_target.cuda(args.gpuid)

            # segment_imgs = segment_imgs.cuda(args.gpuid)
            # input_tags = input_tags.cuda(args.gpuid)

        torch.cuda.synchronize(device=args.gpuid)

        if not args.il_measure:  # go this way
            # ONE
            scores_img2text, scores_seg2text = model(
                input_visuals_source,
                input_visuals_target,
                input_text_source,
                input_text_target,
                #    input_tags,
                #  lengths,
                #    segment_imgs,
            )
            # scores_img2text, scores_img2tag, scores_seg2text, scores_seg2tag
            # import ipdb;ipdb.set_trace()
            loss_img2text = utils.calcul_contraloss(
                args,
                scores_img2text,
                input_visual.size(0),  # bs
                margin,  # 0.2
                max_violation=max_violation,  # False
            )
            # loss_img2tag = utils.calcul_contraloss(
            #         args,
            #         scores_img2tag,
            #         input_visual.size(0), #bs
            #         margin, #0.2
            #         max_violation=max_violation  #False
            # )
            loss_seg2text = utils.calcul_contraloss(
                args,
                scores_seg2text,
                input_visual.size(0),  # bs
                margin,  # 0.2
                max_violation=max_violation,  # False
            )
            # loss_seg2tag = utils.calcul_contraloss(
            #             args,
            #             scores_seg2tag,
            #             input_visual.size(0), #bs
            #             margin, #0.2
            #             max_violation=max_violation  #False
            #     )
            # loss = loss_img2text + loss_img2tag + loss_seg2text + loss_seg2tag
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

        wandb.log(
            {
                "loss": loss.cpu().data.numpy(),
                "loss_img2text": loss_img2text.cpu().data.numpy(),
                # 'loss_img2tag': loss_img2tag.cpu().data.numpy(),
                "loss_seg2text": loss_seg2text.cpu().data.numpy(),
                # 'loss_seg2tag': loss_seg2tag.cpu().data.numpy(),
            }
        )
        optimizer.zero_grad()
        loss.backward()
        # import ipdb;ipdb.set_trace()
        if args.distributed:  # no this way
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses

            train_logger.update("Loss", round(mean_loss.item(), 3))
        else:  # go this way
            if args.il_measure:  # no this way
                train_logger.update("IntraLoss", intra_loss.cpu().data.numpy())

            train_logger.update("Loss", loss.cpu().data.numpy())
            train_logger.update("Loss_img2text", loss_img2text.cpu().data.numpy())
            # train_logger.update('Loss_img2tag', loss_img2tag.cpu().data.numpy())
            train_logger.update("Loss_seg2text", loss_seg2text.cpu().data.numpy())
            # train_logger.update('Loss_seg2tag', loss_seg2tag.cpu().data.numpy())

        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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

            utils.log_to_txt(
                "Epoch [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}\t"
                "{elog}\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    elog=str(train_logger),
                ),
                args.ckpt_save_path + args.model_name + "_" + args.data_name + ".txt",
            )
        # import ipdb;ipdb.set_trace()
        # tb_logger.log_value('epoch', epoch, step=model.Eiters)
        # tb_logger.log_value('step', i, step=model.Eiters)
        # tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        # train_logger.tb_log(tb_logger, step=model.Eiters)

        wandb.log(
            {
                "epoch": epoch,
                "batch_time": batch_time.val,
            }
        )
        train_logger.wandb_log()


def validate(args, val_loader, model):
    print("")
    print("--------------------- Start validation on training set ---------------------")
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
        "Calculate similarity time: {:.2f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n"
        "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n"
        "mR:{:.2f}".format(
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
    print("--------------------- Start validation on training set without sam---------------------")
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
        "Calculate similarity time: {:.2f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n"
        "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n"
        "mR:{:.2f}".format(
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
        "Calculate similarity time: {:.2f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n"
        "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n"
        "mR:{:.2f}".format(
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
    for idx, val_data in enumerate(tqdm(itertools.islice(test_loader, 3))):
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
        "Calculate similarity time: {:.2f} s".format(end - start)
    )  # Print time spent on calculating similarity

    # Calculate accuracy metrics for image-to-text and text-to-image
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # Calculate composite score
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    # Build string representation of all scores
    all_score = (
        "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n"
        "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n"
        "mR:{:.2f}".format(
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
    print("## embedding time: {:.2f} s".format(embed_end - embed_start))

    d = utils.shard_dis_SWAN(
        args, input_visual, input_text, model, lengths=input_text_length
    )

    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))
    print("--------------------- end test ---------------------")
    print("")
    return d


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
