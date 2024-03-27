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
from vocab import deserialize_vocab

def parser_options():
    """
    Parse command line arguments and set hyperparameters for training.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Training path settings
    parser.add_argument('-e', '--experiment_name', default='test', type=str, help="File name for saving checkpoints")
    parser.add_argument('-m', '--model_name', default='SWAN', type=str, help="Model Name")
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

    # SWAN experiment parameters
    parser.add_argument('--sk_1', default=2, type=int, help="Parameter 1 for SWAN experiment")
    parser.add_argument('--sk_2', default=3, type=int, help="Parameter 2 for SWAN experiment")

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

    # Create test data loader
    test_loader = data.get_test_loader_without_sam_mine(args)
    print("len of test_loader is {}".format(len(test_loader)))
    
    # Choose model
    if args.model_name == "SWAN":
        from layers import SWAN as models
    elif args.model_name == "ours":
        from layers import Ours as models
    else:
        raise NotImplementedError

    # Initialize the model
    model = models.factory_without_sam(args, cuda=True, data_parallel=args.distributed)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpuid))
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    
    # Test the model
    rsum_, all_scores_ = engine.validate_test_without_sam(args, test_loader, model)
    print("Test scores:", all_scores_)
     

if __name__ == '__main__':
    args = parser_options()
    main(args)