from ast import arg
import os,random,copy
import shutil
import torch
import argparse
import wandb
from loguru import logger
import torch.distributed as dist
import utils
import data
import engine
import time
from utils.vocab import deserialize_vocab

def parser_options():
    """
    Parse command line arguments and set hyperparameters for training.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Training path settings
    parser.add_argument('-e', '--experiment_name', default='test', type=str, help="File name for saving checkpoints")
    parser.add_argument('-m', '--model_name', default='urbancross', type=str, help="Model Name")
    parser.add_argument('--data_name', default='rsitmd', type=str, help="Dataset Name (e.g., rsitmd or rsicd)")
    parser.add_argument('--data_path', default='./data/', type=str, help="Preprocessed data file path")
    parser.add_argument('--image_path', default='./rs_data/', type=str, help="Remote images data path")
    parser.add_argument('--vocab_path', default='./vocab/', type=str, help="Vocabulary data path")
    parser.add_argument('--resnet_ckpt', default='./aid_28-rsp-resnet-50-ckpt.pth', type=str, help="Path to the ResNet pre-trained model (e.g., aid_28-rsp-resnet-50-ckpt.pth / resnet50-19c8e357.pth)")
    parser.add_argument('--resume', default=False, type=str,help="Path to the pre-trained model for resuming training")
    parser.add_argument('--fix_data', default=False, action='store_true', help='Whether stratified sampling is used')
    parser.add_argument('--step_sample', default=False, action='store_true', help='Whether to use step sampling')
    parser.add_argument('--epochs', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--eval_step', default=1, type=int, help="Evaluation frequency in epochs")
    parser.add_argument('--test_step', default=0, type=int, help="Testing frequency in epochs")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch size for training")
    parser.add_argument('--batch_size_val', default=100, type=int, help="Batch size for validation")
    parser.add_argument('--shard_size', default=256, type=int, help="Batch shard size")
    parser.add_argument('--workers', default=3, type=int, help="Number of workers for data loading")
    parser.add_argument('-kf', '--k_fold_nums', default=1, type=int, help="Total number of k-folds")
    parser.add_argument('--k_fold_current_num', default=0, type=int, help="Current fold number for k-fold validation")

    # Model parameter settings
    parser.add_argument('--embed_dim', default=512, type=int, help="Dimension of the embedding")
    parser.add_argument('--margin', default=0.2, type=float, help="Margin for the triplet loss")
    parser.add_argument('--max_violation', default=False, action='store_true', help="Whether to use max violation in ranking loss")
    parser.add_argument('--grad_clip', default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--il_measure', default=False, help='Similarity measure used (cosine|l1|l2|msd)')
    parser.add_argument('--word_dim', default=300, type=int,help='Dimensionality of the word embedding (e.g., 300, 512)')
    parser.add_argument('--use_bidirectional_rnn', default=True, type=str, help="Whether to use bidirectional RNN")
    parser.add_argument('--is_finetune', default=False, type=str,  help='Whether to finetune ResNet')

    # RNN/GRU model parameters
    parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers')

    # GPU settings
    parser.add_argument('-g', '--gpuid', default=2, type=int, help="GPU device ID to use")
    parser.add_argument('--distributed', default=False, action='store_true', help='Whether to use distributed computing')
    parser.add_argument('--init_method', default='tcp://localhost:18888', help="Initialization method for distributed computing")
    parser.add_argument('--rank', default=0, type=int, help='Rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help="World size")
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="Whether to use mixed precision")

    # Other settings
    parser.add_argument('--logger_name', default='logs/', type=str, help="Path for logging")
    parser.add_argument('-p', '--ckpt_save_path', default='checkpoint/', type=str, help="Path for saving checkpoints")
    parser.add_argument('--print_freq', default=10, type=int,  help="Frequency of printing results")
    parser.add_argument('--lr', default=0.0002, type=float, help="Learning rate")
    parser.add_argument('--lr_update_epoch', default=20, type=int, help="Epochs after which learning rate is updated")
    parser.add_argument('--lr_decay_param', default=0.7, type=float, help="Decay parameter for learning rate")

    # SCAN hyperparameters
    parser.add_argument('--cross_attn', default="t2i", help='t2i|i2t')
    parser.add_argument('--agg_func', default="LogSumExp", help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--lambda_lse', default=6., type=float, help='LogSumExp temperature')
    parser.add_argument('--lambda_softmax', default=9., type=float,  help='Attention softmax temperature')
    parser.add_argument('--raw_feature_norm', default="softmax", help='Normalization method for raw features')

    # WandB settings
    parser.add_argument("--close_wandb", action='store_true', help="Close WandB")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB id")
    parser.add_argument("--wandb_logging_dir", type=str, default='./outputs', help="WandB logging directory")

    # Additional settings
    parser.add_argument("--country", type=str, default='Finland', help="Country name")
    parser.add_argument("--num_seg", type=int, default=10, help="Number of segments")

    args = parser.parse_args()

    # Generate dataset paths
    args.data_path = args.data_path + args.data_name + '_precomp/'
    args.vocab_path = args.vocab_path + args.data_name + '_splits_vocab.json'
    
    # print hyperparameters
    print('-------------------------')
    print('# Hyper Parameters setting')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('-------------------------')
    print('')
    return args


def main(args):
    """
    Main function to train the model.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    # Set WANDB_DIR environment variable
    os.environ["WANDB_DIR"] = args.wandb_logging_dir
    
    # Generate WANDB_ID if not provided
    if not args.wandb_id:
        args.wandb_id = wandb.util.generate_id()
    
    # Login to W&B
    wandb.login(key='d7ec29907ca115fe6d605741c40d09cf563aa0db')
    logger.info(f"W&B ID: {args.wandb_id}")
    
    # Initialize W&B run
    wandb.init(
        project="UrbanCross",
        config=args,
        name=args.experiment_name,
        id=args.wandb_id,
        mode='dryrun',
    )

    # Set random seed
    utils.setup_seed(args.seed)
        
    # Initialize process group for distributed training
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)

    # Choose model
    if args.model_name == "urbancross":
        from layers import urbancross as models
    else:
        raise NotImplementedError

    # Create train, validation and test data loaders
    train_loader, val_loader = data.get_loaders_without_sam_mine(args)
    if args.test_step:
        test_loader = data.get_loaders_without_sam_mine(args)
    print("len of train_loader is {}, len of val_loader is {}".format(len(train_loader), len(val_loader)))

    # Initialize the model
    model = models.factory_without_sam(args, cuda=True, data_parallel=args.distributed)

    # Print and save model information
    if args.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        total_requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_mb = total_params / (1024 * 1024)
        total_requires_grad_params_mb = total_requires_grad_params / (1024 * 1024)

        logger.info("Total Params: {:.2f} MB".format(total_params_mb))
        logger.info("Total Requires_grad Params: {:.2f} MB".format(total_requires_grad_params_mb))
        logger.info(model)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Optionally resume from a checkpoint
    start_epoch = 0
    best_rsum = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpuid))
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'], strict =False)
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(args.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train the Model
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # Adjust learning rate
        utils.adjust_learning_rate(args, optimizer, epoch)

        # Train for one epoch
        engine.train_without_sam(args, train_loader, model, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_step == 0:
            rsum, all_scores = engine.validate_without_sam(args, val_loader, model)
            logger.info("Validation scores: {}".format(all_scores))

            # Save checkpoint if the current model is the best
            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(rsum, best_rsum)

            if args.rank == 0:
                utils.save_checkpoint(
                    {'epoch': epoch + 1, 'model': model.state_dict(), 'best_rsum': best_rsum,'args': args,},
                    epoch,
                    filename = '{}_without_sam_{}_epoch{}_bestRsum{:.4f}.pth'.format(args.data_name, args.model_name, epoch + 1, best_rsum),
                    prefix=args.ckpt_save_path,
                    model_name=args.model_name,
                    args=args,
                )
                logger.info("================ Evaluation result on validation set =====================")
                logger.info("[{}/{}] epochs".format(epoch + 1, args.epochs))
                logger.info("Current validation score:")
                logger.info(all_scores)
                logger.info("Best validation score:")
                logger.info(best_score)

        # Evaluate on test set
        if args.test_step > 0 and (epoch + 1) % args.test_step == 0:
            rsum_, all_scores_ = engine.validate_test_without_sam(args, test_loader, model)
            is_best_ = rsum_ > best_rsum_
            if is_best_:
                best_score_ = all_scores_
            best_rsum_ = max(rsum_, best_rsum_)

            if args.rank == 0:
                logger.info("================ Evaluation result on test set =====================")
                logger.info("[{}/{}] epochs".format(epoch + 1, args.epochs))
                logger.info("Current test score:")
                logger.info(all_scores_)
                logger.info("Best test score:")
                logger.info(best_score_)
              
    if args.distributed:
        # Destroy process group
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser_options()
    main(args)