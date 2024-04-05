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
import os
import shutil

# ==============================================================
def train(args, train_loader, model, optimizer, epoch):

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

    for i, train_data in enumerate(train_loader):
        # images, ids, cap_tokens, segment_imgs, tag_tokens = train_data
        input_visual, ids, input_text, segment_imgs = train_data
        # images, ids, cap_tokens, segment_img, tag_tokens
    
        batch_size = input_visual.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        # input_visual = Variable(images)
        # segment_imgs = Variable(segment_imgs)
        # input_text = Variable(cap_tokens)


        if torch.cuda.is_available():
            input_visual = input_visual.cuda(args.gpuid)
            input_text = input_text.cuda(args.gpuid)
            segment_imgs = segment_imgs.cuda(args.gpuid)
            # input_tags = input_tags.cuda(args.gpuid)
            
        torch.cuda.synchronize(device=args.gpuid)

        if not args.il_measure:  #go this way
            # ONE
            scores_img2text, scores_seg2text = model(
                           input_visual, 
                           input_text, 
                        #    input_tags,
                           #  lengths,
                           segment_imgs,
                           )
            # scores_img2text, scores_img2tag, scores_seg2text, scores_seg2tag
            # import ipdb;ipdb.set_trace()
            loss_img2text = utils.calcul_contraloss(
                        args, 
                        scores_img2text, 
                        input_visual.size(0), #bs
                        margin, #0.2
                        max_violation=max_violation  #False
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
                    input_visual.size(0), #bs
                    margin, #0.2
                    max_violation=max_violation  #False
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
            scores,scores_intra_img,scores_intra_cap = model(input_visual, input_text, lengths)
            intra_loss = utils.calcul_intraloss(args,scores_intra_img) + utils.calcul_intraloss(args,scores_intra_cap)
            loss = utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation) + intra_loss

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)
        # import ipdb; ipdb.set_trace()
        # optimizer_lr = optimizer.param_groups[0]['lr']
        wandb.log(
            {
                'epoch': epoch,
                'loss': loss.cpu().data.numpy(),
                'loss_img2text': loss_img2text.cpu().data.numpy(),
                # 'loss_img2tag': loss_img2tag.cpu().data.numpy(),
                'loss_seg2text': loss_seg2text.cpu().data.numpy(),
                # 'loss_seg2tag': loss_seg2tag.cpu().data.numpy(),
                'lr': optimizer.param_groups[0]['lr'],
            }
        )
        optimizer.zero_grad()
        loss.backward()
        # import ipdb;ipdb.set_trace()
        if args.distributed: #no this way
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses

            train_logger.update('Loss', round(mean_loss.item(),3))
        else: # go this way
            # if args.il_measure: # no this way
            #     train_logger.update('IntraLoss', intra_loss.cpu().data.numpy())
            
            train_logger.update('Loss_avg', loss.cpu().data.numpy())
            train_logger.update('Loss_img2text_avg', loss_img2text.cpu().data.numpy())
            # train_logger.update('Loss_img2tag', loss_img2tag.cpu().data.numpy())
            train_logger.update('Loss_seg2text_avg', loss_seg2text.cpu().data.numpy())
            # train_logger.update('Loss_seg2tag', loss_seg2tag.cpu().data.numpy())
            
        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # import ipdb; ipdb.set_trace()
        #print_freq => 10
        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))
            utils.get_GPU_usage()
            # import ipdb; ipdb.set_trace()
            # utils.log_to_txt(
            #     'Epoch [{0}][{1}/{2}]\t'
            #     'Time {batch_time.val:.3f}\t'
            #     '{elog}\t'
            #         .format(epoch, i, len(train_loader),
            #                 batch_time=batch_time,
            #                 elog=str(train_logger)),
            #     args.ckpt_save_path+ args.model_name + "_" + args.data_name + ".txt"
            # )
        # import ipdb;ipdb.set_trace()
        # tb_logger.log_value('epoch', epoch, step=model.Eiters)
        # tb_logger.log_value('step', i, step=model.Eiters)
        # tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        # train_logger.tb_log(tb_logger, step=model.Eiters)

        wandb.log({
            'epoch': epoch,
            'batch_time': batch_time.val,
        })
        #wandb 记录 loss
        train_logger.wandb_log(epoch)


# ==============================================================
def train_finetune(args, 
                   train_loader_source,
                   train_loader_target,
                   model, 
                   optimizer, 
                   epoch):

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
    # iter_target = iter(train_loader_target)
    
    target_loader_cycle = itertools.cycle(train_loader_target)
    # for i, source_data in enumerate(train_loader_source):
    # for i, (source_data, target_data) in enumerate(zip(train_loader_source, train_loader_target)):
    num_cycle_of_target = -1
    for i, source_data in enumerate(train_loader_source):
    
        images_source, cap_tokens_source = source_data
        # images_target, cap_tokens_target = target_data
        images_target, cap_tokens_target = next(target_loader_cycle)
        # import ipdb; ipdb.set_trace()
        if i % len(train_loader_target) == 0:
            num_cycle_of_target += 1

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
            
            # segment_imgs = segment_imgs.cuda(args.gpuid)
            # input_tags = input_tags.cuda(args.gpuid)
            
        torch.cuda.synchronize(device=args.gpuid)

        # if not args.il_measure:  #go this way
            # ONE
            # import ipdb; ipdb.set_trace()
        clip_loss, adv_loss, filter_ratio  = model(
                       input_visuals_source,
                       input_visuals_target, 
                       input_text_source,
                       input_text_target,
                       num_cycle_of_target = num_cycle_of_target,
                    #    input_tags,
                       #  lengths,
                    #    segment_imgs,
                       )
        loss = clip_loss + adv_loss
            # scores_img2text, scores_img2tag, scores_seg2text, scores_seg2tag
            # import ipdb;ipdb.set_trace()
            # loss_img2text = utils.calcul_contraloss(
            #             args, 
            #             scores_img2text, 
            #             input_visual.size(0), #bs
            #             margin, #0.2
            #             max_violation=max_violation  #False
            #     )
            # loss_img2tag = utils.calcul_contraloss(
            #         args, 
            #         scores_img2tag, 
            #         input_visual.size(0), #bs
            #         margin, #0.2
            #         max_violation=max_violation  #False
            # )
            # loss_seg2text = utils.calcul_contraloss(
            #         args, 
            #         scores_seg2text, 
            #         input_visual.size(0), #bs
            #         margin, #0.2
            #         max_violation=max_violation  #False
            # )
            # loss_seg2tag = utils.calcul_contraloss(
            #             args, 
            #             scores_seg2tag, 
            #             input_visual.size(0), #bs
            #             margin, #0.2
            #             max_violation=max_violation  #False
            #     )
            # loss = loss_img2text + loss_img2tag + loss_seg2text + loss_seg2tag
            # loss = loss_img2text + loss_seg2text
            
        # else:
        #     scores,scores_intra_img,scores_intra_cap = model(input_visual, input_text, lengths)
        #     intra_loss = utils.calcul_intraloss(args,scores_intra_img) + utils.calcul_intraloss(args,scores_intra_cap)
        #     loss = utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation) + intra_loss

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        wandb.log(
            {
                'loss': loss.cpu().data.numpy(),
                'loss_clip': clip_loss.cpu().data.numpy(),
                'loss_adv': adv_loss.cpu().data.numpy(),
            }
        )
        optimizer.zero_grad()
        loss.backward()
        # import ipdb;ipdb.set_trace()
        if args.distributed: #no this way
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses

            train_logger.update('Loss', round(mean_loss.item(),3))
        else: # go this way
            # if args.il_measure: # no this way
            #     train_logger.update('IntraLoss', intra_loss.cpu().data.numpy())
            
            train_logger.update('Loss_avg', loss.cpu().data.numpy())
            # train_logger.update('Loss_img2text', loss_img2text.cpu().data.numpy())
            # # train_logger.update('Loss_img2tag', loss_img2tag.cpu().data.numpy())
            # train_logger.update('Loss_seg2text', loss_seg2text.cpu().data.numpy())
            # train_logger.update('Loss_seg2tag', loss_seg2tag.cpu().data.numpy())
            
        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}(source)][{1}/{3}(target)]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader_source),len(train_loader_target),
                        batch_time=batch_time,
                        elog=str(train_logger))
                )
            logger.info(f'{num_cycle_of_target}_th cycle of target data')
            logger.info(f'filter_ratio: {filter_ratio}')
            utils.get_GPU_usage()
            # utils.log_to_txt(
            #     'Epoch [{0}][{1}/{2}]\t'
            #     'Time {batch_time.val:.3f}\t'
            #     '{elog}\t'
            #         .format(epoch, i, len(train_loader),
            #                 batch_time=batch_time,
            #                 elog=str(train_logger)),
            #     args.ckpt_save_path+ args.model_name + "_" + args.data_name + ".txt"
            # )
        # import ipdb;ipdb.set_trace()
        # tb_logger.log_value('epoch', epoch, step=model.Eiters)
        # tb_logger.log_value('step', i, step=model.Eiters)
        # tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        # train_logger.tb_log(tb_logger, step=model.Eiters)

        wandb.log({
            'epoch': epoch,
            'batch_time': batch_time.val,
        })
        train_logger.wandb_log(epoch)


def train_finetune_curriculum(
                   args, 
                   train_loader_source,
                   train_loader_target,
                   model, 
                   optimizer, 
                   epoch):

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
    # iter_target = iter(train_loader_target)
    
    # 创建 B_loader 的循环迭代器
    target_loader_cycle = itertools.cycle(train_loader_target)
    source_loader_cycle = itertools.cycle(train_loader_source)
    # for i, source_data in enumerate(train_loader_source):
    # for i, (source_data, target_data) in enumerate(zip(train_loader_source, train_loader_target)):
    num_cycle_of_target = 0
    i = 0
    # for i, source_data in enumerate(train_loader_source):
    while num_cycle_of_target <= 4:
    
        # images_source, cap_tokens_source = source_data
        images_source, cap_tokens_source = next(source_loader_cycle)
        # images_target, cap_tokens_target = target_data
        images_target, cap_tokens_target = next(target_loader_cycle)
        # import ipdb; ipdb.set_trace()
        if i % len(train_loader_target) == 0 and i != 0:
            num_cycle_of_target += 1
        if num_cycle_of_target > 4:
            break
        
        i += 1

    # 在这里添加你的代码
        # images, ids, cap_tokens, segment_imgs, tag_tokens = train_data
        # images, ids, cap_tokens, segment_imgs = train_data
        # images_source, cap_tokens_source = source_data
        # print(i)
        # try:
        #     target_data = next(iter_target)
        #     # import ipdb; ipdb.set_trace()
            
        #     # if len(target_data) == 3:
        #     images_target, cap_tokens_target = target_data
        #     # else:
        #     #     raise ValueError("Expected target_data to have 3 elements, but got {}".format(len(target_data)))
            
        # except StopIteration:
        #     iter_target = iter(train_loader_target)
        #     images_target, cap_tokens_target = next(iter_target)
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

        # if not args.il_measure:  #go this way
            # ONE
            # import ipdb; ipdb.set_trace()
        clip_loss, adv_loss, filter_ratio  = model(
                       input_visuals_source,
                       input_visuals_target, 
                       input_text_source,
                       input_text_target,
                       num_cycle_of_target = num_cycle_of_target,
                    #    input_tags,
                       #  lengths,
                    #    segment_imgs,
                       )
        loss = clip_loss + adv_loss
            # scores_img2text, scores_img2tag, scores_seg2text, scores_seg2tag
            # import ipdb;ipdb.set_trace()
            # loss_img2text = utils.calcul_contraloss(
            #             args, 
            #             scores_img2text, 
            #             input_visual.size(0), #bs
            #             margin, #0.2
            #             max_violation=max_violation  #False
            #     )
            # loss_img2tag = utils.calcul_contraloss(
            #         args, 
            #         scores_img2tag, 
            #         input_visual.size(0), #bs
            #         margin, #0.2
            #         max_violation=max_violation  #False
            # )
            # loss_seg2text = utils.calcul_contraloss(
            #         args, 
            #         scores_seg2text, 
            #         input_visual.size(0), #bs
            #         margin, #0.2
            #         max_violation=max_violation  #False
            # )
            # loss_seg2tag = utils.calcul_contraloss(
            #             args, 
            #             scores_seg2tag, 
            #             input_visual.size(0), #bs
            #             margin, #0.2
            #             max_violation=max_violation  #False
            #     )
            # loss = loss_img2text + loss_img2tag + loss_seg2text + loss_seg2tag
            # loss = loss_img2text + loss_seg2text
            
        # else:
        #     scores,scores_intra_img,scores_intra_cap = model(input_visual, input_text, lengths)
        #     intra_loss = utils.calcul_intraloss(args,scores_intra_img) + utils.calcul_intraloss(args,scores_intra_cap)
        #     loss = utils.calcul_contraloss(args, scores, input_visual.size(0), margin, max_violation=max_violation) + intra_loss

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        wandb.log(
            {
                'loss': loss.cpu().data.numpy(),
                # 'loss_img2text': loss_img2text.cpu().data.numpy(),
                # # 'loss_img2tag': loss_img2tag.cpu().data.numpy(),
                # 'loss_seg2text': loss_seg2text.cpu().data.numpy(),
                # 'loss_seg2tag': loss_seg2tag.cpu().data.numpy(),
            }
        )
        optimizer.zero_grad()
        loss.backward()
        # import ipdb;ipdb.set_trace()
        if args.distributed: #no this way
            loss = utils.reduce_value(args, loss, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses

            train_logger.update('Loss', round(mean_loss.item(),3))
        else: # go this way
            # if args.il_measure: # no this way
            #     train_logger.update('IntraLoss', intra_loss.cpu().data.numpy())
            
            train_logger.update('Loss_avg', loss.cpu().data.numpy())
            # train_logger.update('Loss_img2text', loss_img2text.cpu().data.numpy())
            # # train_logger.update('Loss_img2tag', loss_img2tag.cpu().data.numpy())
            # train_logger.update('Loss_seg2text', loss_seg2text.cpu().data.numpy())
            # train_logger.update('Loss_seg2tag', loss_seg2tag.cpu().data.numpy())
            
        torch.cuda.synchronize(device=args.gpuid)
        optimizer.step()
        torch.cuda.synchronize(device=args.gpuid)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and args.rank == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}(source)][{1}/{3}(target)]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader_source),len(train_loader_target),
                        batch_time=batch_time,
                        elog=str(train_logger))
                )
            logger.info(f'{num_cycle_of_target}_th cycle of target data')
            logger.info(f'filter_ratio: {filter_ratio}')
            utils.get_GPU_usage()
            # utils.log_to_txt(
            #     'Epoch [{0}][{1}/{2}]\t'
            #     'Time {batch_time.val:.3f}\t'
            #     '{elog}\t'
            #         .format(epoch, i, len(train_loader),
            #                 batch_time=batch_time,
            #                 elog=str(train_logger)),
            #     args.ckpt_save_path+ args.model_name + "_" + args.data_name + ".txt"
            # )
        # import ipdb;ipdb.set_trace()
        # tb_logger.log_value('epoch', epoch, step=model.Eiters)
        # tb_logger.log_value('step', i, step=model.Eiters)
        # tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        # train_logger.tb_log(tb_logger, step=model.Eiters)

        wandb.log({
            'epoch': epoch,
            'batch_time': batch_time.val,
        })
        train_logger.wandb_log(epoch)


def validate(args, val_loader, model, epoch):
    # print('')
    logger.info("--------------------- start val on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    
    # # input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    # input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))

    # # input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    # input_text = np.zeros((len(val_loader.dataset), 77), dtype=np.int64)
    # # input_text_length = [0] * len(val_loader.dataset)

    input_visual = []
    input_text = []
    input_seg = []
    # scores = []
    # for i, val_data in enumerate(val_loader):
    for idx, val_data in enumerate(val_loader):
        # images, captions, lengths, ids = val_data
        images, ids, cap_tokens, segment_img = val_data
        input_visual.append(images)
        input_text.append(cap_tokens)
        input_seg.append(segment_img)
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)
    input_seg = torch.cat(input_seg, dim=0)
    # import ipdb; ipdb.set_trace()
    #     images = images.cuda(args.gpuid)
    #     cap_tokens = cap_tokens.cuda(args.gpuid)
    #     segment_img = segment_img.cuda(args.gpuid)
    #     with torch.no_grad():
    #         score_img2text, score_seg2text = model(images, cap_tokens, segment_img)
    #     scores.append(score_img2text)
    #     logger.info(
    #             f'Eval [{i}/{len(val_loader)}]\t'
    #             # 'Time {batch_time.val:.3f}\t'
    #             # '{elog}\t'
    #             # .format(i, len(val_loader),
    #             #         # batch_time=batch_time,
    #             #         # elog=str(train_logger)
    #             #         )
    #     )
    #     # import ipdb;ipdb.set_trace()
    #     # score_img2text, score_seg2text = model(images,cap_tokens,segment_img)
    #     # # for (id, img, cap, l) in zip(ids, (images.numpy().copy()),  (captions.numpy().copy()), lengths):
    #     # # for (id, img, cap) in zip(ids, (images.numpy().copy()),  (cap_tokens.numpy().copy())):
    #     # for (id, img, cap, seg) in zip(ids, images,  cap_tokens, segment_img):
        
    #     #     # input_visual = input_visual.cuda(args.gpuid)
    #     #     # input_text = input_text.cuda(args.gpuid)
    #     #     # input_seg = input_seg.cuda(args.gpuid)
    #     #     img = img.cuda(args.gpuid)
    #     #     cap = cap.cuda(args.gpuid)
    #     #     seg = seg.cuda(args.gpuid)
    #     #     import ipdb; ipdb.set_trace()
    #     #     score_img2text, score_seg2text = model(img,cap,seg)
    #     #     scores.append(score_img2text)
    #     #     # # input_visual[id] = img
    #     #     # input_visual.append(img)
    #     #     # # input_text[id, :captions.size(1)] = cap
    #     #     # # input_text[id] = cap
    #     #     # input_text.append(cap)
    #     #     # # print(id)
    #     #     # # input_text_length[id] = l
    #     #     # input_seg.append(seg)
    # # scores = torch.stack(scores, dim=0)
    # input_visual = torch.cat(input_visual, dim=0)
    # input_text = torch.cat(input_text, dim=0)
    # input_seg = torch.cat(input_seg, dim=0)
    
    
    # import ipdb; ipdb.set_trace()
    # res = model(input_visual, input_text, input_seg)
    # input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    # d = utils.shard_dis_MSSF(args, input_visual, input_text, model,
    #                          lengths=input_text_lengeth)
    d = utils.shard_dis_mine(
                             args, 
                             input_visual, 
                             input_text, 
                             input_seg,
                             model,
                            #  lengths=input_text_length
                            )
    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))

    #image to text
    # (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    #text to image
    # (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # import ipdb; ipdb.set_trace()
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    logger.info("--------------------- end val on training ---------------------")
    # print('')

    # tb_logger.log_value('r1i', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i', r10i, step=model.Eiters)
    # tb_logger.log_value('medri', medri, step=model.Eiters)
    # tb_logger.log_value('meanri', meanri, step=model.Eiters)
    # tb_logger.log_value('r1t', r1t, step=model.Eiters)
    # tb_logger.log_value('r5t', r5t, step=model.Eiters)
    # tb_logger.log_value('r10t', r10t, step=model.Eiters)
    # tb_logger.log_value('medrt', medrt, step=model.Eiters)
    # tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    # tb_logger.log_value('rsum', currscore, step=model.Eiters)

    wandb.log({
        'epoch': epoch,
        'val/r1i': r1i,
        'val/r5i': r5i,
        'val/r10i': r10i,
        'val/medri': medri,
        'val/meanri': meanri,
        'val/r1t': r1t,
        'val/r5t': r5t,
        'val/r10t': r10t,
        'val/medrt': medrt,
        'val/meanrt': meanrt,
        'val/rsum': currscore
    })
    
    return currscore, all_score


def validate_finetune(
                      args, 
                      val_loader, 
                      model, 
                      epoch
                      ):
    # print('')
    logger.info("--------------------- start val on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    
    # # input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    # input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))

    # # input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    # input_text = np.zeros((len(val_loader.dataset), 77), dtype=np.int64)
    # # input_text_length = [0] * len(val_loader.dataset)

    input_visual = []
    input_text = []
    # input_seg = []
    # scores = []
    # for i, val_data in enumerate(val_loader):
    for idx, val_data in enumerate(val_loader):
        # images, captions, lengths, ids = val_data
        images, cap_tokens = val_data
        input_visual.append(images)
        input_text.append(cap_tokens)
        # input_seg.append(segment_img)
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)
    # input_seg = torch.cat(input_seg, dim=0)
    # import ipdb; ipdb.set_trace()
    #     images = images.cuda(args.gpuid)
    #     cap_tokens = cap_tokens.cuda(args.gpuid)
    #     segment_img = segment_img.cuda(args.gpuid)
    #     with torch.no_grad():
    #         score_img2text, score_seg2text = model(images, cap_tokens, segment_img)
    #     scores.append(score_img2text)
    #     logger.info(
    #             f'Eval [{i}/{len(val_loader)}]\t'
    #             # 'Time {batch_time.val:.3f}\t'
    #             # '{elog}\t'
    #             # .format(i, len(val_loader),
    #             #         # batch_time=batch_time,
    #             #         # elog=str(train_logger)
    #             #         )
    #     )
    #     # import ipdb;ipdb.set_trace()
    #     # score_img2text, score_seg2text = model(images,cap_tokens,segment_img)
    #     # # for (id, img, cap, l) in zip(ids, (images.numpy().copy()),  (captions.numpy().copy()), lengths):
    #     # # for (id, img, cap) in zip(ids, (images.numpy().copy()),  (cap_tokens.numpy().copy())):
    #     # for (id, img, cap, seg) in zip(ids, images,  cap_tokens, segment_img):
        
    #     #     # input_visual = input_visual.cuda(args.gpuid)
    #     #     # input_text = input_text.cuda(args.gpuid)
    #     #     # input_seg = input_seg.cuda(args.gpuid)
    #     #     img = img.cuda(args.gpuid)
    #     #     cap = cap.cuda(args.gpuid)
    #     #     seg = seg.cuda(args.gpuid)
    #     #     import ipdb; ipdb.set_trace()
    #     #     score_img2text, score_seg2text = model(img,cap,seg)
    #     #     scores.append(score_img2text)
    #     #     # # input_visual[id] = img
    #     #     # input_visual.append(img)
    #     #     # # input_text[id, :captions.size(1)] = cap
    #     #     # # input_text[id] = cap
    #     #     # input_text.append(cap)
    #     #     # # print(id)
    #     #     # # input_text_length[id] = l
    #     #     # input_seg.append(seg)
    # # scores = torch.stack(scores, dim=0)
    # input_visual = torch.cat(input_visual, dim=0)
    # input_text = torch.cat(input_text, dim=0)
    # input_seg = torch.cat(input_seg, dim=0)
    
    
    # import ipdb; ipdb.set_trace()
    # res = model(input_visual, input_text, input_seg)
    # input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    # d = utils.shard_dis_MSSF(args, input_visual, input_text, model,
    #                          lengths=input_text_lengeth)
    d = utils.shard_dis_mine_finetune(args, 
                             input_visual, 
                             input_text, 
                            #  input_seg,
                             model,
                            #  lengths=input_text_length
                            )
    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))

    #image to text
    # (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)
    #text to image
    # (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    # import ipdb; ipdb.set_trace()
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    logger.info("--------------------- end val on training ---------------------")
    # print('')

    # tb_logger.log_value('r1i', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i', r10i, step=model.Eiters)
    # tb_logger.log_value('medri', medri, step=model.Eiters)
    # tb_logger.log_value('meanri', meanri, step=model.Eiters)
    # tb_logger.log_value('r1t', r1t, step=model.Eiters)
    # tb_logger.log_value('r5t', r5t, step=model.Eiters)
    # tb_logger.log_value('r10t', r10t, step=model.Eiters)
    # tb_logger.log_value('medrt', medrt, step=model.Eiters)
    # tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    # tb_logger.log_value('rsum', currscore, step=model.Eiters)

    wandb.log({
        'epoch': epoch,
        'val/r1i': r1i,
        'val/r5i': r5i,
        'val/r10i': r10i,
        'val/medri': medri,
        'val/meanri': meanri,
        'val/r1t': r1t,
        'val/r5t': r5t,
        'val/r10t': r10t,
        'val/medrt': medrt,
        'val/meanrt': meanrt,
        'val/rsum': currscore
    })
    
    return currscore, all_score


def validate_test(args, test_loader, model):
    print('')
    print("--------------------- start test on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    # input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))


    # input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    # input_text_length = [0] * len(test_loader.dataset)
    input_visual = []
    input_text = []
    input_seg = []
    
    
    # embed_start = time.time()
    # for i, val_data in enumerate(test_loader):
    for idx, val_data in enumerate(tqdm(test_loader, 3)):
        # images, captions, lengths, ids = val_data
        images, ids, cap_tokens, segment_img = val_data
        input_visual.append(images)
        input_text.append(cap_tokens)
        input_seg.append(segment_img)
        
        # for (id, img,cap, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), lengths):
        
        #     input_visual[id] = img

        #     input_text[id, :captions.size(1)] = cap
        #     input_text_length[id] = l
    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)
    input_seg = torch.cat(input_seg, dim=0)
    # input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    # embed_end = time.time()
    # print("## test embedding time: {:.2f} s".format(embed_end-embed_start))

    # d = utils.shard_dis_MSSF(args, input_visual, input_text, model, lengths=input_text_length)
    d = utils.shard_dis_mine(args, 
                             input_visual, 
                             input_text, 
                             input_seg,
                             model,
                            #  lengths=input_text_length
                            )
    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))

    # (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)

    # (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    print("--------------------- end test on training ---------------------")
    print('')

    # tb_logger.log_value('r1i_test', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i_test', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i_test', r10i, step=model.Eiters)
    # tb_logger.log_value('medri_test', medri, step=model.Eiters)
    # tb_logger.log_value('meanri_test', meanri, step=model.Eiters)
    # tb_logger.log_value('r1t_test', r1t, step=model.Eiters)
    # tb_logger.log_value('r5t_test', r5t, step=model.Eiters)
    # tb_logger.log_value('r10t_test', r10t, step=model.Eiters)
    # tb_logger.log_value('medrt_test', medrt, step=model.Eiters)
    # tb_logger.log_value('meanrt_test', meanrt, step=model.Eiters)
    # tb_logger.log_value('rsum_test', currscore, step=model.Eiters)
    
    wandb.log({
        'test/r1i': r1i,
        'test/r5i': r5i,
        'test/r10i': r10i,
        'test/medri': medri,
        'test/meanri': meanri,
        'test/r1t': r1t,
        'test/r5t': r5t,
        'test/r10t': r10t,
        'test/medrt': medrt,
        'test/meanrt': meanrt,
        'test/rsum': currscore
    })
    return currscore, all_score


def test(args, test_loader, model):
    print('')
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

        for (id, img,cap, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), lengths):
            input_visual[id] = img

            input_text[id, :captions.size(1)] = cap
            input_text_length[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    embed_end = time.time()
    print("## embedding time: {:.2f} s".format(embed_end-embed_start))

    d = utils.shard_dis_SWAN(args, 
                             input_visual, 
                             input_text, 
                             model, 
                             lengths=input_text_length
                             )

    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))
    print("--------------------- end test ---------------------")
    print('')
    return d

def test_mine(args, test_loader, model):
    print('')
    print("--------------------- start test on training ---------------------")
    model.eval()

    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    # input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))


    # input_text = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
    # input_text_length = [0] * len(test_loader.dataset)
    input_visual = []
    input_text = []
    img_paths = []
    captions = []
    # input_seg = []
        # input_seg = []
    # input_visual = torch.empty((len(test_loader.dataset), 3,224,224))
    # input_text = torch.empty((len(test_loader.dataset), 77))

    # import ipdb;ipdb.set_trace()
    # embed_start = time.time()
    # for i, val_data in enumerate(test_loader):
    #len(test_loader是402)
    
    # for idx, val_data in enumerate(tqdm(test_loader)):
    for idx, val_data in enumerate(tqdm(itertools.islice(test_loader, 20))):
        # images, captions, lengths, ids = val_data
        # images, ids, cap_tokens, segment_img = val_data
        images, cap_tokens, img_path, caption = val_data
        input_visual.append(images)
        input_text.append(cap_tokens)
        img_paths.extend(img_path)
        captions.extend(caption)
        # input_seg.append(segment_img)


    input_visual = torch.cat(input_visual, dim=0)
    input_text = torch.cat(input_text, dim=0)
    # input_seg = torch.cat(input_seg, dim=0)
    # input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    # import ipdb; ipdb.set_trace()
    # input_visual = torch.cat(input_visual, dim=0)
    # input_text = torch.cat(input_text, dim=0)
    # input_seg = torch.cat(input_seg, dim=0)
    # input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    # import ipdb; ipdb.set_trace()
    # embed_end = time.time()
    # print("## test embedding time: {:.2f} s".format(embed_end-embed_start))
    logger.info('begin to compute distance')
    # d = utils.shard_dis_MSSF(args, input_visual, input_text, model, lengths=input_text_length)
    d = utils.shard_dis_mine_finetune(
                             args, 
                             input_visual, 
                             input_text, 
                            #  input_seg,
                             model,
                            #  lengths=input_text_length
                            )
    # Normalize d to [0, 1]
    d_normalized = (d - np.min(d)) / (np.max(d) - np.min(d))

    
    for i in tqdm(range(len(img_paths))):
        top10_indices = np.argsort(-d_normalized[i])[:10]
        top10_captions = [captions[idx] for idx in top10_indices]
        top10_values = [d_normalized[i][idx] for idx in top10_indices]
        
        savepath = os.path.join(args.ckpt_save_path, img_paths[i].split('/')[-1])
        os.makedirs(savepath, exist_ok=True)
        shutil.copy(img_paths[i], savepath)
        # import ipdb;ipdb.set_trace()
        with open(os.path.join(savepath, 'top10_captions.txt'), 'w') as f:
            for j in range(10):
                f.write(f'{top10_captions[j]}\n')
                f.write(f'{top10_values[j]}\n')
                f.write('\n')
    
    
    
    end = time.time()
    print("calculate similarity time: {:.2f} s".format(end - start))
    

    # (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t_mine(d)

    # (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2t_mine(d.T)

    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "i2t => r1i:{:.2f} r5i:{:.2f} r10i:{:.2f} medri:{:.2f} meanri:{:.2f}\n" \
                "t2i => r1t:{:.2f} r5t:{:.2f} r10t:{:.2f} medrt:{:.2f} meanrt:{:.2f}\n" \
                "mR:{:.2f}".format(r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore)

    print("--------------------- end test on training ---------------------")
    print('')

    # tb_logger.log_value('r1i_test', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i_test', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i_test', r10i, step=model.Eiters)
    # tb_logger.log_value('medri_test', medri, step=model.Eiters)
    # tb_logger.log_value('meanri_test', meanri, step=model.Eiters)
    # tb_logger.log_value('r1t_test', r1t, step=model.Eiters)
    # tb_logger.log_value('r5t_test', r5t, step=model.Eiters)
    # tb_logger.log_value('r10t_test', r10t, step=model.Eiters)
    # tb_logger.log_value('medrt_test', medrt, step=model.Eiters)
    # tb_logger.log_value('meanrt_test', meanrt, step=model.Eiters)
    # tb_logger.log_value('rsum_test', currscore, step=model.Eiters)
    
    wandb.log({
        'test/r1i': r1i,
        'test/r5i': r5i,
        'test/r10i': r10i,
        'test/medri': medri,
        'test/meanri': meanri,
        'test/r1t': r1t,
        'test/r5t': r5t,
        'test/r10t': r10t,
        'test/medrt': medrt,
        'test/meanrt': meanrt,
        'test/rsum': currscore
    })
    return currscore, all_score


def save(args, test_loader, model):
    print('')
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

        for (id, img,cap, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), lengths):
            input_visual[id] = img

            input_text[id, :captions.size(1)] = cap
            input_text_length[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    img_emb, text_emb = utils.save_img_text_emb(args, input_visual, input_text, model, lengths=input_text_length)

    return img_emb, text_emb