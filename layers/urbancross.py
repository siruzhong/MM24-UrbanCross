import math
import open_clip_mine as open_clip
import numpy as np
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.init
from torch.nn.parallel.distributed import DistributedDataParallel
import copy
import torch
import torch.nn as nn
import torch.nn.init
from .resnet import resnet50
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# MODEL_NAME = "ViT-L-14"
# PRETRAINED = "laion2B-s32B-b82K"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2B-s34B-b88K"

class UrbanCross(nn.Module):
    def __init__(self, args):
        """
        Initialize the UrbanCross model.

        Args:
            args: Model configuration arguments.
        """
        super().__init__()
        # Create OpenCLIP model and transforms
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME,  
            pretrained=PRETRAINED,
            output_dict=True,
        )
        # Create a copy of the OpenCLIP model for segmented images
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        # Remove the transformer layer from the copied model
        del self.clip_img_seg.transformer

    def forward(self, img, text, segment_imgs):
        """
        Forward pass of the UrbanCross model.

        Args:
            img (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.
            segment_imgs (torch.Tensor): Input segmented images tensor.

        Returns:
            torch.Tensor: Similarity scores between image and text.
            torch.Tensor: Similarity scores between segmented images and text.
        """
        with torch.cuda.amp.autocast():
            # Get features for the input image and text
            clip_model_out = self.clip_model(img, text)
            img_emb = clip_model_out["image_features"]
            text_emb = clip_model_out["text_features"]

            # Get the number of segments
            num_seg = segment_imgs.shape[0]
            seg_emb_list = []
            
            # Flatten the segment_imgs tensor
            bs, num_seg, _, _, _ = segment_imgs.shape
            segment_imgs_reshaped = segment_imgs.view(bs * num_seg, 3, 224, 224)
            
            # Encode segmented images
            img_seg_emb = self.clip_img_seg.encode_image(segment_imgs_reshaped)
            img_seg_emb = img_seg_emb.view(bs, num_seg, -1)
            # Calculate the feature mean of each batch
            img_seg_emb = img_seg_emb.mean(dim=1)

            # Calculate cosine similarity between image and text
            sim_img2text = cosine_sim(img_emb, text_emb)
            # Calculate cosine similarity between segmented images and text
            sim_seg2text = cosine_sim(img_seg_emb, text_emb)

        return sim_img2text, sim_seg2text
    

class UrbanCross_without_sam(nn.Module):
    def __init__(self, args):
        """
        Initialize the UrbanCross model.

        Args:
            args: Model configuration arguments.
        """
        super().__init__()
        # Create OpenCLIP model and transforms
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME,
            pretrained=PRETRAINED,  # mscoco_finetuned_laion2B-s13B-b90k
            output_dict=True,
        )

    def forward(self, img, text):
        """
        Forward pass of the UrbanCross model.

        Args:
            img (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.

        Returns:
            torch.Tensor: Similarity scores between image and text.
            torch.Tensor: Similarity scores between segmented images and text.
        """
        with torch.cuda.amp.autocast():
            # Get features for the input image and text
            clip_model_out = self.clip_model(img, text)
            img_emb = clip_model_out["image_features"]
            text_emb = clip_model_out["text_features"]

            # Calculate cosine similarity between image and text
            sim_img2text = cosine_sim(img_emb, text_emb)

        return sim_img2text


class AdversarialLoss(nn.Module):
    def __init__(self):
        """
        Initialize the AdversarialLoss module.
        """
        super(AdversarialLoss, self).__init__()
        # self.W_tilde_2 = 1.0

    def forward(self, model, source, target, W2):
        """
        Forward pass of the AdversarialLoss module.

        Args:
            model (nn.Module): Discriminator model.
            F_s_tilde (torch.Tensor): Source features tensor.
            F_t_tilde (torch.Tensor): Target features tensor.
            W2 (torch.Tensor): Weight tensor.

        Returns:
            torch.Tensor: Adversarial loss value.
        """
        # Calculate the discriminator's probability on source features
        prob_source = model(source)
        # Calculate the discriminator's probability on target features
        prob_target = model(target)

        # Ensure the discriminator output is in the range [0, 1] by applying sigmoid
        prob_source = torch.sigmoid(prob_source)
        prob_target = torch.sigmoid(prob_target)

        # Expand the weight tensor to match the shape of probability tensors
        W2 = W2.unsqueeze(dim=1)
        
        # Calculate the adversarial loss using weighted cross-entropy loss
        loss = -(
            torch.mean(W2 * torch.log(prob_source)) +
            torch.mean(W2 * torch.log(1 - prob_target))
        )
        
        # Note: Negative sign is used because we typically minimize the loss, and the original equation is for maximization
        return loss


class UrbanCross_finetune(nn.Module):
    def __init__(self, args):
        """
        Initialize the UrbanCross_finetune model.

        Args:
            args: Model configuration arguments.
        """
        super().__init__()
        # Create OpenCLIP model and transforms
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME,
            pretrained=PRETRAINED,
            output_dict=True,
        )
        # Create a copy of the OpenCLIP model for segmented images
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        # Remove the transformer layer from the copied model
        del self.clip_img_seg.transformer
       
        # Define the discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2),  # Output layer with 2 units for binary classification
        )

        # Initialize the adversarial loss module
        self.adv_loss = AdversarialLoss()

        # Initialize the CLIP loss module
        self.clip_loss = open_clip.ClipLoss()

    def forward(self, img_source, img_target, text_source, text_target, num_cycle_of_target=0, val=False):
        if val:
            return self.forward_val(img_target, text_target)
        
        ratio = 0.2

        with torch.cuda.amp.autocast():
            clip_model_out_source = self.clip_model(img_source, text_source)
            clip_model_out_target = self.clip_model(img_target, text_target)
            
            img_emb_source = clip_model_out_source['image_features']
            img_emb_target = clip_model_out_target['image_features']
            text_emb_source = clip_model_out_source['text_features']
            text_emb_target = clip_model_out_target['text_features']
            
            # Calculate similarity between text embeddings
            W1 = cosine_sim(text_emb_target, text_emb_source)
            W1_mean = W1.mean(dim=0)
            batchsize = img_emb_source.shape[0]
            selected_batchsize = int(batchsize * ratio)
            # Sort W1 along each row, from large to small
            sorted_W1, _ = torch.sort(W1, dim=1, descending=True)
            W2 = sorted_W1[:, :selected_batchsize]
            _, sorted_W1_mean_index = torch.sort(W1_mean, descending=True)

            img_emb_source_filtered = img_emb_source[sorted_W1_mean_index[:selected_batchsize]]
            text_emb_source_filtered = text_emb_source[sorted_W1_mean_index[:selected_batchsize]]

            # Sum W_2 over the second dimension to get a vector
            W2 = torch.sum(W2, dim=1)

            # Scale W_tilde_2 to range [0, 1]
            W2_min = torch.min(W2)
            W2_max = torch.max(W2)
            W2 = (W2 - W2_min) / (W2_max - W2_min)

            # Normalize W_tilde_2 to sum to 1
            W2 = W2 / torch.sum(W2)
            
            # Calculate triplet loss
            adv_loss_img = self.adv_loss(
                                        self.discriminator, 
                                        img_emb_source_filtered, 
                                        img_emb_target, 
                                        W2,
                                    )
            adv_loss_text = self.adv_loss(
                                        self.discriminator, 
                                        text_emb_source_filtered, 
                                        text_emb_target, 
                                        W2,
                                    )
            adv_loss = adv_loss_img + adv_loss_text
            
            clip_loss = self.clip_loss(img_emb_source_filtered, 
                                    text_emb_source_filtered,
                                    logit_scale=1.0
                                    #    img_emb_target, 
                                    #    text_emb_target
                                    )
        return clip_loss, adv_loss, ratio
    
    
    def forward_val(self, img_target, text_target):
    
        with torch.cuda.amp.autocast():
            clip_model_out = self.clip_model(img_target, text_target)
    
            
            img_emb = clip_model_out['image_features']
            text_emb = clip_model_out['text_features']
            
            sim_img2text = cosine_sim(img_emb, text_emb)
        
        return sim_img2text
    

class UrbanCross_finetune_curriculum(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME,
            pretrained=PRETRAINED,
            output_dict=True,
        )

        self.clip_img_seg = copy.deepcopy(self.clip_model)
        del self.clip_img_seg.transformer
        self.discriminator = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2),
        )
        
        self.adv_loss = AdversarialLoss()
        self.clip_loss = open_clip.ClipLoss()

    def forward(self, img_source, img_target, text_source, text_target, num_cycle_of_target=0, val=False):
        if val:
            return self.forward_val(img_target, text_target)
        ratio = 0.2

        with torch.cuda.amp.autocast():
            clip_model_out_source = self.clip_model(img_source, text_source)
            clip_model_out_target = self.clip_model(img_target, text_target)
            
            img_emb_source = clip_model_out_source['image_features']
            img_emb_target = clip_model_out_target['image_features']
            
            text_emb_source = clip_model_out_source['text_features']
            text_emb_target = clip_model_out_target['text_features']
            
            # Calculate similarity between text embeddings
            W1 = cosine_sim(text_emb_target, text_emb_source)
            W1_mean = W1.mean(dim=0)

            batchsize = img_emb_source.shape[0]
            selected_batchsize = int(batchsize * ratio)
   
            # Sort W1 along each row, from large to small
            sorted_W1, _ = torch.sort(W1, dim=1, descending=True)
          
            W2 = sorted_W1[:, selected_batchsize * num_cycle_of_target : selected_batchsize * (num_cycle_of_target+1)]
            _, sorted_W1_mean_index = torch.sort(W1_mean, descending=True)

            img_emb_source_filtered = img_emb_source[sorted_W1_mean_index[:selected_batchsize]]
            text_emb_source_filtered = text_emb_source[sorted_W1_mean_index[:selected_batchsize]]

            # Sum W_2 over the second dimension to get a vector
            W2 = torch.sum(W2, dim=1)

            # Scale W_tilde_2 to range [0, 1]
            W2_min = torch.min(W2)
            W2_max = torch.max(W2)
            W2 = (W2 - W2_min) / (W2_max - W2_min)

            # Normalize W_tilde_2 to sum to 1
            W2 = W2 / torch.sum(W2)
            # Calculate triplet loss
            # triplet_loss = nn.TripletMarginLoss()
            # loss = triplet_loss(img_emb_source, text_emb_source, topk_W1)

            adv_loss = self.adv_loss(self.discriminator, 
                                    img_emb_source_filtered, 
                                    img_emb_target, 
                                    W2
                                    )
            clip_loss = self.clip_loss(img_emb_source_filtered, 
                                    text_emb_source_filtered,
                                    logit_scale=1.0
                                    )
        return clip_loss, adv_loss, ratio
       
    def forward_val(self, img_target, text_target):
        with torch.cuda.amp.autocast():
            clip_model_out = self.clip_model(img_target, text_target)
            
            img_emb = clip_model_out['image_features']
            text_emb = clip_model_out['text_features']
            
            sim_img2text = cosine_sim(img_emb, text_emb)
        
        return sim_img2text
        
    
class UrbanCross_zeroshot(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME,
            pretrained=PRETRAINED,
            output_dict=True,
        )
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        del self.clip_img_seg.transformer

    def forward(self, img, text):
        with torch.cuda.amp.autocast():
            clip_model_out = self.clip_model(img, text)
            
            img_emb = clip_model_out['image_features']
            text_emb = clip_model_out['text_features']

            # Calculating similarity
            sim_img2text = cosine_sim(img_emb, text_emb)

        return sim_img2text


#====================
# Some Reuse Function
#====================
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12

def clones(module, N):
    """Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def factory(args, cuda=True, data_parallel=False):
    args_new = copy.copy(args)

    model_without_ddp = UrbanCross(args_new)

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(model, device_ids=[args.gpuid],find_unused_parameters=False)
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp


def factory_without_sam(args, cuda=True, data_parallel=False):
    """
    Factory function to create and initialize the model.

    Args:
        args: Namespace containing model configuration and parameters.
        cuda (bool, optional): Flag indicating whether to use CUDA (GPU). Defaults to True.
        data_parallel (bool, optional): Flag indicating whether to use data parallelism. Defaults to False.

    Returns:
        nn.Module: Initialized model instance.
    """
    # Create a copy of args to avoid modifying the original object
    args_new = copy.copy(args)

    # Initialize the model without DistributedDataParallel (DDP)
    model_without_ddp = UrbanCross_without_sam(
        args_new,
    )

    # Move the model to GPU if cuda is True
    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    # Apply data parallelism if data_parallel is True
    if data_parallel:
        # Convert BatchNorm layers to SyncBatchNorm for distributed training
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        # Initialize DistributedDataParallel with model and GPU device ID
        model = DistributedDataParallel(
            model, device_ids=[args.gpuid], find_unused_parameters=False
        )
        # Get the module attribute of the model, as DDP wraps the model with an additional layer
        model_without_ddp = model.module
        # Ensure CUDA is enabled if data parallelism is used
        if not cuda:
            raise ValueError

    return model_without_ddp


def factory_with_finetune(args, cuda=True, data_parallel=False):
    args_new = copy.copy(args)

    model_without_ddp = UrbanCross_finetune(args_new)

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(model, device_ids=[args.gpuid],find_unused_parameters=False)
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp


def factory_finetune(args, cuda=True, data_parallel=False):
    args_new = copy.copy(args)

    model_without_ddp = UrbanCross_finetune(args_new)

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(model, device_ids=[args.gpuid],find_unused_parameters=False)
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp


def factory_finetune_curriculum(args, cuda=True, data_parallel=False):
    args_new = copy.copy(args)

    model_without_ddp = UrbanCross_finetune_curriculum(args_new)

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(
            model, device_ids=[args.gpuid], find_unused_parameters=False
        )
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp


def factory_zeroshot(
                     args, 
                    #  word2idx, 
                     cuda=True, 
                     data_parallel=False
                     ):
    args_new = copy.copy(args)

    # model_without_ddp = SWAN(args_new, word2idx)
    model_without_ddp = UrbanCross_zeroshot(args_new, 
                                            # word2idx
                                            )

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(model, device_ids=[args.gpuid],find_unused_parameters=False)
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp