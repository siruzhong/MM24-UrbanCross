import os,random,copy
import shutil
import torch
import argparse
import wandb
from loguru import logger
import torch.distributed as dist
import utils.utils as utils
import data
import engine
from utils.vocab import deserialize_vocab


def parser_options():
    parser = argparse.ArgumentParser()

    # Training path settings
    parser.add_argument('-e', '--experiment_name', default='test', type=str, help="the file name of ckpt save")
    parser.add_argument('-m', '--model_name', default='urbancross', type=str, help="Model Name")
    parser.add_argument('--data_name', default='rsitmd', type=str, help="Dataset Name.(eg.rsitmd or rsicd)")
    parser.add_argument('--data_path', default='./data/', type=str, help="Preprocessed data file path")
    parser.add_argument('--image_path', default='./rs_data/', type=str, help="remote images data path")
    parser.add_argument('--vocab_path', default='./vocab/', type=str, help="vocab data path")
    parser.add_argument('--resnet_ckpt', default='./aid_28-rsp-resnet-50-ckpt.pth', type=str, help="restnet pre model path.eg.(aid_28-rsp-resnet-50-ckpt.pth / resnet50-19c8e357.pth)")
    parser.add_argument('--resume', default=False, type=str, help="the pre-trained model path")
    parser.add_argument('--fix_data', default=False, action='store_true', help='Whether stratified sampling is used')
    parser.add_argument('--step_sample', default=False, action='store_true', help='Whether stratified sampling is used')
    parser.add_argument('--epochs', default=100, type=int, help="the epochs of train")
    parser.add_argument('--eval_step', default=1, type=int, help="the epochs of eval")
    parser.add_argument('--test_step', default=0, type=int, help="the epochs of test")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch train size")
    parser.add_argument('--batch_size_source', default=100, type=int, help="Batch train size")
    parser.add_argument('--batch_size_target', default=100, type=int, help="Batch train size")
    parser.add_argument('--batch_size_val_source', default=100, type=int, help="Batch val size")
    parser.add_argument('--batch_size_val_target', default=100, type=int, help="Batch val size")
    parser.add_argument('--shard_size', default=256, type=int, help="Batch shard size")
    parser.add_argument('--workers', default=3, type=int, help="the worker num of dataloader")
    parser.add_argument('-kf', '--k_fold_nums', default=1, type=int, help="the total num of k_flod")
    parser.add_argument('--k_fold_current_num', default=0, type=int, help="current num of k_fold")

    # Model parameter settings
    parser.add_argument('--embed_dim', default=512, type=int, help="the embedding's dim")
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--max_violation', default=False, action='store_true')
    parser.add_argument('--grad_clip', default=0.0, type=float)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--il_measure', default=False, help='Similarity measure used (cosine|l1|l2|msd)')

    # RNN/GRU model parameter
    parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.(e.g. 300, 512)')
    parser.add_argument('--use_bidirectional_rnn', default=True, type=str)
    parser.add_argument('--is_finetune', default=False, type=str, help='Finetune resnet or not')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers.')

    # GPU setting
    parser.add_argument('-g', '--gpuid', default=2, type=int, help="which gpu to use")
    parser.add_argument('--distributed', default=False, action='store_true', help='Whether to use parallel computing')
    parser.add_argument('--init_method', default='tcp://localhost:18888', help="init-method")
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help="world size")
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="whether to use mix precision")

    # No set setting
    parser.add_argument('--logger_name', default='logs/', type=str, help="the path of logs")
    parser.add_argument('-p', '--ckpt_save_path', default='checkpoint_fix_data/', type=str, help="the path of checkpoint save")
    parser.add_argument('--print_freq', default=10, type=int, help="Print result frequency")
    parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
    parser.add_argument('--lr_update_epoch', default=20, type=int, help="the update epoch of learning rate")
    parser.add_argument('--lr_decay_param', default=0.7, type=float, help="the decay_param of learning rate")

    # SCAN hyperparameters
    parser.add_argument('--cross_attn', default="t2i", help='t2i|i2t')
    parser.add_argument('--agg_func', default="LogSumExp", help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--lambda_lse', default=6., type=float, help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float, help='Attention softmax temperature.')
    parser.add_argument('--raw_feature_norm', default="softmax", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')

    parser.add_argument("--close_wandb", action='store_true',)
    parser.add_argument("--wandb_id", type=str, default=None,)
    parser.add_argument("--wandb_logging_dir", type=str, default='./outputs',)
    parser.add_argument("--country", type=str,)
    parser.add_argument("--country_source", type=str, default='Finland',)
    parser.add_argument("--country_target", type=str, default='Finland',)
    parser.add_argument("--load_path", type=str,)

    args = parser.parse_args()

    # Generate dataset path
    args.data_path = args.data_path + args.data_name + '_precomp/'
    args.vocab_path = args.vocab_path + args.data_name + '_splits_vocab.json'

    # Print hyperparameters
    print('-------------------------')
    print('# Hyper Parameters setting')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('-------------------------')
    print('')

    return args


def main(args):
    # Set WANDB_DIR environment variable
    os.environ["WANDB_DIR"] = args.wandb_logging_dir
    
    # Generate WANDB_ID if not provided
    if not args.wandb_id:
        args.wandb_id = wandb.util.generate_id()
    
    # Login to W&B
    wandb.login(key='d7ec29907ca115fe6d605741c40d09cf563aa0db')
    logger.info(f"W&B ID: {args.wandb_id}")
    
    # Initialize Weights and Biases run
    wandb.init(
        project="UrbanCross",
        config=args,
        name=args.experiment_name,
        id=args.wandb_id,
        mode='dryrun',
    )

    # Set random seed for reproducibility
    utils.setup_seed(args.seed)

    # Initialize distributed training if enabled
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)

    # Choose the model
    if args.model_name == "urbancross" or args.model_name == "urbancross_finetune":
        from layers import urbancross as models
    else:
        raise NotImplementedError
    
    logger.info(args)
    # Create dataset, model, criterion, and optimizer
    train_loader_source, train_loader_target, train_dataset_source, train_dataset_target, val_loader_target, val_dataset_target = data.get_loaders_finetune(
        args,
    )
    logger.info(f"len of train_set is {len(train_dataset_source)}(source)/{len(train_dataset_target)}(target)")
    logger.info(f"len of val_set is {len(val_dataset_target)}(target)")

    # Load pretrained model weights
    model = models.factory_with_finetune(args, cuda=True, data_parallel=args.distributed)
    pretrained_weight = torch.load(args.load_path, map_location='cuda:{}'.format(args.gpuid))
    model.load_state_dict(pretrained_weight['model'], strict=False)

    # Print and save model info
    if args.rank == 0:        
        total_params = sum(p.numel() for p in model.parameters())
        total_requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_mb = total_params / (1024 * 1024)
        total_requires_grad_params_mb = total_requires_grad_params / (1024 * 1024)
        logger.info("Total Params: {:.2f} MB".format(total_params_mb))
        logger.info("Total Requires_grad Params: {:.2f} MB".format(total_requires_grad_params_mb))
        logger.info(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # optionally resume from a checkpoint
    start_epoch = 0
    best_rsum = 0
    best_rsum_ = 0
    best_score = ""
    best_score_ = ""
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpuid))
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'], strict =False)
            # Eiters is used to show logs as the continuation of another
            model.Eiters = checkpoint['Eiters']
   
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(args.resume, start_epoch, best_rsum))
            rsum, all_scores = engine.validate(args, val_loader, model)
            print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train the Model
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        utils.adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        engine.train_finetune(args, train_loader_source, train_loader_target, model, optimizer, epoch)

        # Evaluate on validation set
        if (epoch + 1) % args.eval_step == 0:
            rsum, all_scores = engine.validate_finetune(args, val_loader_target, model)

            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(rsum, best_rsum)

            if args.rank == 0:
                logger.info("================ evaluate result on val set =====================")
                logger.info("Current =>[{}/{}] epochs".format(epoch + 1, args.epochs))
                logger.info("Now val score:")
                logger.info(all_scores)
                logger.info("Best val score:")
                logger.info(best_score)
                logger.info("=================================================================")
 
                utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'best_rsum': best_rsum,
                        'args': args,
                    },
                    filename=f'ckpt_{args.model_name}_{epoch}_{best_rsum:.2f}.pth',
                    prefix=args.ckpt_save_path,
                    model_name=args.model_name
                )

    if args.distributed:
        # destroy process
        dist.destroy_process_group()

if __name__ == '__main__':
    args = parser_options()
    main(args)
