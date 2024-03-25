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
# import ipdb;ipdb.set_trace()
import sys
sys.path.append('..')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from loguru import logger
# from torch.cuda.amp import autocast
#============
# Main Model
#============
class UrbanCross(nn.Module):
    def __init__(self, args, 
                        # word2idx
                 ):
        super().__init__()
        # self.Eiters = 0
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            # model_name="coca_ViT-L-14", 
            model_name="ViT-L-14",
            pretrained='laion2B-s32B-b82K',  #mscoco_finetuned_laion2B-s13B-b90k
            output_dict=True,
        )
        # self.clip_model1 = copy.deepcopy(self.clip_model)
        # self.clip_model2 = copy.deepcopy(self.clip_model)
        # self.clip_model3 = copy.deepcopy(self.clip_model)
        # del self.clip_model1.visual
        # del self.clip_model2.visual
        # del self.clip_model3.visual
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        del self.clip_img_seg.transformer
        # self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # import ipdb;ipdb.set_trace()
        # Image Encoder
        # self.image_encoder =  ImageExtractFeature(args)
        # # Text Encoder
        # self.text_encoder = TextExtractFeature(args, word2idx)
        # # Scene Fine-Grained Sensing Module
        # self.sfgs = SFGS(args)
        # # Vsion Global-Local Features Fusion
        # self.agg = Aggregation(args)
        # # Text Coarse-Grained Enhancement Module
        # self.tcge = TCGE(args)

        # self.sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        # # sam.to(device=device)

        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def forward(self, 
                img , 
                text, 
                # input_tags,
                # lengths,
                segment_imgs,
                # images, ids, cap_tokens, segment_img, tag_tokens
                ):
        #img [bs,3,256,256]
        # text是[bs,30]
        with torch.cuda.amp.autocast():
            # import ipdb;ipdb.set_trace()
            # title = copy.deepcopy(text)
            # ingredients = copy.deepcopy(text)
            # instructions = copy.deepcopy(text)
            clip_model_out = self.clip_model(img, text)
            
            # title_emb = self.clip_model1.encode_text(title)
            # ingredients_emb = self.clip_model2.encode_text(ingredients)
            # instructions_emb = self.clip_model3.encode_text(instructions)
            # tags_emb = torch.cat((title_emb,ingredients_emb,instructions_emb), dim=1)
            
            # tags_emb = self.clip_model1.encode_text(input_tags)
            
            # ipdb> clip_model_out.keys()
            # dict_keys(['image_features', 'text_features', 'logit_scale'])
            
            # ipdb> clip_model_out['image_features'].shape
            # torch.Size([100, 768])
            # ipdb> clip_model_out['text_features'].shape
            # torch.Size([100, 768])
            img_emb = clip_model_out['image_features']
            text_emb = clip_model_out['text_features']
            # Visual Part
            # vl_fea, vg_emb = self.image_encoder(img)
            # #vl_fea [bs,32,64,64]
            # #vg_emb [bs,512]
            num_seg = segment_imgs.shape[0]
            seg_emb_list = []
            # import ipdb; ipdb.set_trace()
            
            # segment_imgs [bs,num_seg,3,224,224]
            bs, num_seg, _, _, _ = segment_imgs.shape
            # 改变形状为 [bs * num_seg, 3, 224, 224]
            segment_imgs_reshaped = segment_imgs.view(bs * num_seg, 3, 224, 224)
            img_seg_emb = self.clip_img_seg.encode_image(segment_imgs_reshaped)
            img_seg_emb = img_seg_emb.view(bs, num_seg, -1)
            # 计算每个批次的特征均值
            img_seg_emb = img_seg_emb.mean(dim=1)

            
            # import ipdb;ipdb.set_trace()
            # for i in range(num_seg):
            #     seg_emb_list.append(self.clip_img_seg.encode_image(segment_imgs[i]))
            # img_seg_emb = torch.mean(torch.stack(seg_emb_list,dim=0), dim=0)
            # # img_seg_emb = self.clip_img_seg.encode_image(segment_imgs)
            # #scene fine-grained sensing module
            # vl_emb = self.sfgs(vl_fea)
            # #vl_emb [bs,512]
            # import ipdb;ipdb.set_trace()
            # img_emb = self.agg(vl_emb, vg_emb)
            # #img_emb [bs,512]

            # # Textual Part
            # cap_fea = self.text_encoder(text, lengths)
            # #cap_fea [bs,21,512]
            # list_of_imgs = torch.chunk(img, img.shape[0], dim=0)

            # # 使用squeeze函数将每个元素的维度为1的维度去除，得到形状为[3, h, w]
            # list_of_imgs = [{'image':t.squeeze(0),
            #                  'original_size':(t.shape[-2],t.shape[-1])} for t in list_of_imgs]
            # import ipdb;ipdb.set_trace()
            # self.sam(list_of_imgs, multimask_output=False)
            # masks = self.mask_generator.generate(img.permute(0,2,3,1))
            # #textual coarse-grained enhancement module
            # text_emb = self.tcge(cap_fea, lengths)
            # #text_emb [bs,512]

            # Calculating similarity
            sim_img2text = cosine_sim(img_emb, text_emb)
            # sim_img2tag = cosine_sim(img_emb, tags_emb)
            sim_seg2text = cosine_sim(img_seg_emb, text_emb)
            # sim_seg2tag = cosine_sim(img_seg_emb, tags_emb)
            # sims = cosine_sim(img_emb, text_emb)
            #sims [bs,bs]
            # import ipdb; ipdb.set_trace()
            # return sims
            # return sim_img2text, sim_img2tag, sim_seg2text, sim_seg2tag
        return sim_img2text, sim_seg2text

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.W_tilde_2 = 1.0 
    
    def forward(self, model, F_s_tilde, F_t_tilde, W2):
        # Calculate the discriminator's probability on source features
        prob_source = model(F_s_tilde)
        # Calculate the discriminator's probability on target features
        prob_target = model(F_t_tilde)
        
        # Make sure the discriminator output is in the range [0,1]
        #prob_source  [bs,2]
        prob_source = torch.sigmoid(prob_source)
        #prob_target [bs,2]
        prob_target = torch.sigmoid(prob_target)
        # Calculate the loss
        # loss = - self.W_tilde_2 * (torch.mean(torch.log(prob_source)) + 
        #                      torch.mean(torch.log(1 - prob_target)))
        W2 = W2.unsqueeze(dim=1)
        # import ipdb; ipdb.set_trace()
        loss = - (torch.mean(W2 * torch.log(prob_source)) + 
                     torch.mean(W2 * torch.log(1 - prob_target)))
        return loss  # The negative sign is used because we typically minimize the loss, and the original equation is for maximization


class UrbanCross_finetune(nn.Module):
    def __init__(self, 
                 args, 
                #  word2idx
                 ):
        super().__init__()
        # self.Eiters = 0
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            # model_name="coca_ViT-L-14", 
            model_name="ViT-L-14",
            pretrained='laion2B-s32B-b82K',  #mscoco_finetuned_laion2B-s13B-b90k
            output_dict=True,
        )
        # self.clip_model1 = copy.deepcopy(self.clip_model)
        # self.clip_model2 = copy.deepcopy(self.clip_model)
        # self.clip_model3 = copy.deepcopy(self.clip_model)
        # del self.clip_model1.visual
        # del self.clip_model2.visual
        # del self.clip_model3.visual
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        del self.clip_img_seg.transformer
        # self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # import ipdb;ipdb.set_trace()
        # Image Encoder
        # self.image_encoder =  ImageExtractFeature(args)
        # # Text Encoder
        # self.text_encoder = TextExtractFeature(args, word2idx)
        # # Scene Fine-Grained Sensing Module
        # self.sfgs = SFGS(args)
        # # Vsion Global-Local Features Fusion
        # self.agg = Aggregation(args)
        # # Text Coarse-Grained Enhancement Module
        # self.tcge = TCGE(args)
        self.discriminator = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2),
            # nn.Sigmoid()
        )
        
        self.adv_loss = AdversarialLoss()
        # self.sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        # # sam.to(device=device)
        self.clip_loss = open_clip.ClipLoss()
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def forward(self, 
                img_source,
                img_target, 
                text_source,
                text_target,
                num_cycle_of_target=0,
                # input_tags,
                # lengths,
                # segment_imgs,
                # images, ids, cap_tokens, segment_img, tag_tokens
                val=False
                ):
        #img [bs,3,256,256]
        # text是[bs,30]
        if val:
            return self.forward_val(img_target, text_target)
        # import ipdb;ipdb.set_trace()
        # title = copy.deepcopy(text)
        # ingredients = copy.deepcopy(text)
        # instructions = copy.deepcopy(text)
        ratio = 0.5
        # ratio = 0.5 - 0.1 * num_cycle_of_target
        # logger.info(f'ratio: {ratio}')
        with torch.cuda.amp.autocast():
            clip_model_out_source = self.clip_model(img_source, text_source)
            clip_model_out_target = self.clip_model(img_target, text_target)
            
            
            # tags_emb = self.clip_model1.encode_text(input_tags)
            
            # ipdb> clip_model_out.keys()
            # dict_keys(['image_features', 'text_features', 'logit_scale'])
            
            # ipdb> clip_model_out['image_features'].shape
            # torch.Size([100, 768])
            # ipdb> clip_model_out['text_features'].shape
            # torch.Size([100, 768])
            img_emb_source = clip_model_out_source['image_features']
            img_emb_target = clip_model_out_target['image_features']
            
            text_emb_source = clip_model_out_source['text_features']
            text_emb_target = clip_model_out_target['text_features']
            
            
            # Calculate similarity between text embeddings
            W1 = cosine_sim(text_emb_target, text_emb_source)
            # W1 = cosine_sim(text_emb_source, text_emb_target).mean(dim=1)
            W1_mean = W1.mean(dim=0)
            # import ipdb; ipdb.set_trace()
            batchsize = img_emb_source.shape[0]
            
            selected_batchsize = int(batchsize * ratio)
            # selected_batchsize = batchsize
            # Sort W1 along each row
            # 从大到小排序
            sorted_W1, _ = torch.sort(W1, dim=1, descending=True)
            # import ipdb; ipdb.set_trace()
            # Select top k values from each row
            W2 = sorted_W1[:, :selected_batchsize]
            _, sorted_W1_mean_index = torch.sort(W1_mean, descending=True)
            # import ipdb; ipdb.set_trace()
            # img_emb_source_sorted = img_emb_source[sort_indices[:,:selected_batchsize]]
            img_emb_source_filtered = img_emb_source[sorted_W1_mean_index[:selected_batchsize]]
            text_emb_source_filtered = text_emb_source[sorted_W1_mean_index[:selected_batchsize]]
            
            # img_emb_source_sorted = torch.index_select(img_emb_source, dim=0, index=sort_indices[:, :selected_batchsize])
            # 通过索引操作取出 text1 中相似度高的项
            # top_text1_items = torch.index_select(text1, dim=0, index=sorted_indices[:, :bs2])


            # Sum W_2 over the second dimension to get a vector
            W2 = torch.sum(W2, dim=1)

            # Scale W_tilde_2 to range [0, 1]
            W2_min = torch.min(W2)
            W2_max = torch.max(W2)
            W2 = (W2 - W2_min) / (W2_max - W2_min)

            # Normalize W_tilde_2 to sum to 1
            # W2 = selected_batchsize * W2 / torch.sum(W2)
            W2 = W2 / torch.sum(W2)
            # Calculate triplet loss
            # triplet_loss = nn.TripletMarginLoss()
            # loss = triplet_loss(img_emb_source, text_emb_source, topk_W1)
        
        
            # self.discriminator(text_emb_source)
            # import ipdb; ipdb.set_trace()
            adv_loss = self.adv_loss(self.discriminator, 
                                    img_emb_source_filtered, 
                                    img_emb_target, 
                                    W2
                                    )
            clip_loss = self.clip_loss(img_emb_source_filtered, 
                                    text_emb_source_filtered,
                                    logit_scale=1.0
                                    #    img_emb_target, 
                                    #    text_emb_target
                                    )
        return clip_loss, adv_loss, ratio
        # import ipdb; ipdb.set_trace()
        # num_seg = segment_imgs.shape[0]
        # seg_emb_list = []
        # # import ipdb; ipdb.set_trace()
        
        # # segment_imgs [bs,num_seg,3,224,224]
        # bs, num_seg, _, _, _ = segment_imgs.shape
        # # 改变形状为 [bs * num_seg, 3, 224, 224]
        # segment_imgs_reshaped = segment_imgs.view(bs * num_seg, 3, 224, 224)
        # img_seg_emb = self.clip_img_seg.encode_image(segment_imgs_reshaped)
        # img_seg_emb = img_seg_emb.view(bs, num_seg, -1)
        # # 计算每个批次的特征均值
        # img_seg_emb = img_seg_emb.mean(dim=1)


        # # import ipdb;ipdb.set_trace()
        # # for i in range(num_seg):
        # #     seg_emb_list.append(self.clip_img_seg.encode_image(segment_imgs[i]))
        # # img_seg_emb = torch.mean(torch.stack(seg_emb_list,dim=0), dim=0)
        # # # img_seg_emb = self.clip_img_seg.encode_image(segment_imgs)
        # # #scene fine-grained sensing module
        # # vl_emb = self.sfgs(vl_fea)
        # # #vl_emb [bs,512]
        # # import ipdb;ipdb.set_trace()
        # # img_emb = self.agg(vl_emb, vg_emb)
        # # #img_emb [bs,512]

        # # # Textual Part
        # # cap_fea = self.text_encoder(text, lengths)
        # # #cap_fea [bs,21,512]
        # # list_of_imgs = torch.chunk(img, img.shape[0], dim=0)

        # # # 使用squeeze函数将每个元素的维度为1的维度去除，得到形状为[3, h, w]
        # # list_of_imgs = [{'image':t.squeeze(0),
        # #                  'original_size':(t.shape[-2],t.shape[-1])} for t in list_of_imgs]
        # # import ipdb;ipdb.set_trace()
        # # self.sam(list_of_imgs, multimask_output=False)
        # # masks = self.mask_generator.generate(img.permute(0,2,3,1))
        # # #textual coarse-grained enhancement module
        # # text_emb = self.tcge(cap_fea, lengths)
        # # #text_emb [bs,512]

        # # Calculating similarity
        # sim_img2text = cosine_sim(img_emb, text_emb)
        # # sim_img2tag = cosine_sim(img_emb, tags_emb)
        # sim_seg2text = cosine_sim(img_seg_emb, text_emb)
        # # sim_seg2tag = cosine_sim(img_seg_emb, tags_emb)
        # # sims = cosine_sim(img_emb, text_emb)
        # #sims [bs,bs]
        # # import ipdb; ipdb.set_trace()
        # # return sims
        # return sim_img2text, sim_img2tag, sim_seg2text, sim_seg2tag
        # return loss
    
    def forward_val(
                self, 
                # img_source,
                img_target, 
                # text_source,
                text_target,
                # input_tags,
                # lengths,
                # segment_imgs,
                # images, ids, cap_tokens, segment_img, tag_tokens
                ):
        #img [bs,3,256,256]
        # text是[bs,30]
        
        # import ipdb;ipdb.set_trace()
        # title = copy.deepcopy(text)
        # ingredients = copy.deepcopy(text)
        # instructions = copy.deepcopy(text)
        # ratio = 0.5
        with torch.cuda.amp.autocast():
            clip_model_out = self.clip_model(img_target, text_target)
            
            # tags_emb = self.clip_model1.encode_text(input_tags)
            
            # ipdb> clip_model_out.keys()
            # dict_keys(['image_features', 'text_features', 'logit_scale'])
            
            # ipdb> clip_model_out['image_features'].shape
            # torch.Size([100, 768])
            # ipdb> clip_model_out['text_features'].shape
            # torch.Size([100, 768])
            # img_emb_source = clip_model_out_source['image_features']
            # img_emb_target = clip_model_out_target['image_features']
            
            # text_emb_source = clip_model_out_source['text_features']
            # text_emb_target = clip_model_out_target['text_features']
            
            img_emb = clip_model_out['image_features']
            text_emb = clip_model_out['text_features']
            
            sim_img2text = cosine_sim(img_emb, text_emb)
        
        return sim_img2text
        #     # Calculate similarity between text embeddings
        #     W1 = cosine_sim(text_emb_target, text_emb_source)
        #     # W1 = cosine_sim(text_emb_source, text_emb_target).mean(dim=1)
        #     W1_mean = W1.mean(dim=0)
        #     # import ipdb; ipdb.set_trace()
        #     batchsize = img_emb_source.shape[0]
            
        #     selected_batchsize = int(batchsize * ratio)
        #     # selected_batchsize = batchsize
        #     # Sort W1 along each row
        #     # 从大到小排序
        #     sorted_W1, _ = torch.sort(W1, dim=1, descending=True)
        #     # import ipdb; ipdb.set_trace()
        #     # Select top k values from each row
        #     W2 = sorted_W1[:, :selected_batchsize]
        #     _, sorted_W1_mean_index = torch.sort(W1_mean, descending=True)
        #     # import ipdb; ipdb.set_trace()
        #     # img_emb_source_sorted = img_emb_source[sort_indices[:,:selected_batchsize]]
        #     img_emb_source_filtered = img_emb_source[sorted_W1_mean_index[:selected_batchsize]]
        #     text_emb_source_filtered = text_emb_source[sorted_W1_mean_index[:selected_batchsize]]
            
        #     # img_emb_source_sorted = torch.index_select(img_emb_source, dim=0, index=sort_indices[:, :selected_batchsize])
        #     # 通过索引操作取出 text1 中相似度高的项
        #     # top_text1_items = torch.index_select(text1, dim=0, index=sorted_indices[:, :bs2])


        #     # Sum W_2 over the second dimension to get a vector
        #     W2 = torch.sum(W2, dim=1)

        #     # Scale W_tilde_2 to range [0, 1]
        #     W2_min = torch.min(W2)
        #     W2_max = torch.max(W2)
        #     W2 = (W2 - W2_min) / (W2_max - W2_min)

        #     # Normalize W_tilde_2 to sum to 1
        #     # W2 = selected_batchsize * W2 / torch.sum(W2)
        #     W2 = W2 / torch.sum(W2)
        #     # Calculate triplet loss
        #     # triplet_loss = nn.TripletMarginLoss()
        #     # loss = triplet_loss(img_emb_source, text_emb_source, topk_W1)
        
        
        #     # self.discriminator(text_emb_source)
        #     # import ipdb; ipdb.set_trace()
        #     adv_loss = self.adv_loss(self.discriminator, 
        #                             img_emb_source_filtered, 
        #                             img_emb_target, 
        #                             W2
        #                             )
        #     clip_loss = self.clip_loss(img_emb_source_filtered, 
        #                             text_emb_source_filtered,
        #                             logit_scale=1.0
        #                             #    img_emb_target, 
        #                             #    text_emb_target
        #                             )
        # return clip_loss, adv_loss 
    
class UrbanCross_wo_seg(nn.Module):
    def __init__(self, args, 
                        # word2idx
                 ):
        super().__init__()
        # self.Eiters = 0
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            # model_name="coca_ViT-L-14", 
            model_name="ViT-L-14",
            pretrained='laion2B-s32B-b82K',  #mscoco_finetuned_laion2B-s13B-b90k
            output_dict=True,
        )
        # self.clip_model1 = copy.deepcopy(self.clip_model)
        # self.clip_model2 = copy.deepcopy(self.clip_model)
        # self.clip_model3 = copy.deepcopy(self.clip_model)
        # del self.clip_model1.visual
        # del self.clip_model2.visual
        # del self.clip_model3.visual
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        del self.clip_img_seg.transformer
        # self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # import ipdb;ipdb.set_trace()
        # Image Encoder
        # self.image_encoder =  ImageExtractFeature(args)
        # # Text Encoder
        # self.text_encoder = TextExtractFeature(args, word2idx)
        # # Scene Fine-Grained Sensing Module
        # self.sfgs = SFGS(args)
        # # Vsion Global-Local Features Fusion
        # self.agg = Aggregation(args)
        # # Text Coarse-Grained Enhancement Module
        # self.tcge = TCGE(args)

        # self.sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        # # sam.to(device=device)

        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def forward(self, 
                img , 
                text, 
                # input_tags,
                # lengths,
                # segment_imgs,
                # images, ids, cap_tokens, segment_img, tag_tokens
                ):
        #img [bs,3,256,256]
        # text是[bs,30]
        with torch.cuda.amp.autocast():
            # import ipdb;ipdb.set_trace()
            # title = copy.deepcopy(text)
            # ingredients = copy.deepcopy(text)
            # instructions = copy.deepcopy(text)
            clip_model_out = self.clip_model(img, text)
            
            # title_emb = self.clip_model1.encode_text(title)
            # ingredients_emb = self.clip_model2.encode_text(ingredients)
            # instructions_emb = self.clip_model3.encode_text(instructions)
            # tags_emb = torch.cat((title_emb,ingredients_emb,instructions_emb), dim=1)
            
            # tags_emb = self.clip_model1.encode_text(input_tags)
            
            # ipdb> clip_model_out.keys()
            # dict_keys(['image_features', 'text_features', 'logit_scale'])
            
            # ipdb> clip_model_out['image_features'].shape
            # torch.Size([100, 768])
            # ipdb> clip_model_out['text_features'].shape
            # torch.Size([100, 768])
            img_emb = clip_model_out['image_features']
            text_emb = clip_model_out['text_features']
            # Visual Part
            # vl_fea, vg_emb = self.image_encoder(img)
            # #vl_fea [bs,32,64,64]
            # #vg_emb [bs,512]
            # num_seg = segment_imgs.shape[0]
            # seg_emb_list = []
            # import ipdb; ipdb.set_trace()
            
            # segment_imgs [bs,num_seg,3,224,224]
            # bs, num_seg, _, _, _ = segment_imgs.shape
            # # 改变形状为 [bs * num_seg, 3, 224, 224]
            # segment_imgs_reshaped = segment_imgs.view(bs * num_seg, 3, 224, 224)
            # img_seg_emb = self.clip_img_seg.encode_image(segment_imgs_reshaped)
            # img_seg_emb = img_seg_emb.view(bs, num_seg, -1)
            # # 计算每个批次的特征均值
            # img_seg_emb = img_seg_emb.mean(dim=1)

            
            # import ipdb;ipdb.set_trace()
            # for i in range(num_seg):
            #     seg_emb_list.append(self.clip_img_seg.encode_image(segment_imgs[i]))
            # img_seg_emb = torch.mean(torch.stack(seg_emb_list,dim=0), dim=0)
            # # img_seg_emb = self.clip_img_seg.encode_image(segment_imgs)
            # #scene fine-grained sensing module
            # vl_emb = self.sfgs(vl_fea)
            # #vl_emb [bs,512]
            # import ipdb;ipdb.set_trace()
            # img_emb = self.agg(vl_emb, vg_emb)
            # #img_emb [bs,512]

            # # Textual Part
            # cap_fea = self.text_encoder(text, lengths)
            # #cap_fea [bs,21,512]
            # list_of_imgs = torch.chunk(img, img.shape[0], dim=0)

            # # 使用squeeze函数将每个元素的维度为1的维度去除，得到形状为[3, h, w]
            # list_of_imgs = [{'image':t.squeeze(0),
            #                  'original_size':(t.shape[-2],t.shape[-1])} for t in list_of_imgs]
            # import ipdb;ipdb.set_trace()
            # self.sam(list_of_imgs, multimask_output=False)
            # masks = self.mask_generator.generate(img.permute(0,2,3,1))
            # #textual coarse-grained enhancement module
            # text_emb = self.tcge(cap_fea, lengths)
            # #text_emb [bs,512]

            # Calculating similarity
            sim_img2text = cosine_sim(img_emb, text_emb)
            # sim_img2tag = cosine_sim(img_emb, tags_emb)
            # sim_seg2text = cosine_sim(img_seg_emb, text_emb)
            # sim_seg2tag = cosine_sim(img_seg_emb, tags_emb)
            # sims = cosine_sim(img_emb, text_emb)
            #sims [bs,bs]
            # import ipdb; ipdb.set_trace()
            # return sims
            # return sim_img2text, sim_img2tag, sim_seg2text, sim_seg2tag
        return sim_img2text

# #=========================
# # Image feature extraction
# #========================
# class ImageExtractFeature(nn.Module):
#     def __init__(self, args):
#         super(ImageExtractFeature, self).__init__()
#         self.embed_dim = args.embed_dim
#         self.is_finetune = args.is_finetune
#         # load resnet50
#         self.resnet = resnet50(args, num_classes = 30,pretrained=True)
#         # Vision Multi-Scale Fusion Module
#         self.vmsf = VMSF(args)

#         # Filtering local features
#         self.conv_filter = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True))

#         for param in self.resnet.parameters():
#             param.requires_grad = self.is_finetune

#     def forward(self, img):
#         # Shallow features
#         x = self.resnet.conv1(img)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         # Deep features
#         deep_fea_1 = self.resnet.layer2(self.resnet.layer1(x))
#         deep_fea_2 = self.resnet.layer3(deep_fea_1)
#         deep_fea_3 = self.resnet.layer4(deep_fea_2)

#         shallow_fea = self.conv_filter(x)

#         deep_feas = (deep_fea_1, deep_fea_2, deep_fea_3)
#         vg_emb = self.vmsf(deep_feas)
#         return shallow_fea, vg_emb

# # =========================
# # Text feature extraction
# # =========================
# class TextExtractFeature(nn.Module):
#     def __init__(self, args, word2idx):
#         super(TextExtractFeature, self).__init__()
#         self.gpuid = args.gpuid
#         self.embed_dim = args.embed_dim
#         self.word_dim = args.word_dim
#         self.vocab_size = 8590
#         self.num_layers = args.num_layers
#         self.use_bidirectional_rnn = args.use_bidirectional_rnn
#         # word embedding
#         self.embed = nn.Embedding(self.vocab_size, self.word_dim)

#         # caption embedding
#         self.use_bidirectional_rnn = self.use_bidirectional_rnn
#         print('=> using bidirectional rnn:{}'.format(self.use_bidirectional_rnn))
#         self.rnn = nn.GRU(self.word_dim, self.embed_dim, self.num_layers,
#                           batch_first=True, bidirectional=self.use_bidirectional_rnn)
#         self.projection = nn.Linear(self.embed_dim, self.embed_dim)

#         self.dropout = nn.Dropout(0.4)

#         self.init_weights(word2idx, self.word_dim)

#     def init_weights(self, word2idx, word_dim):
#         # Load pretrained word embedding
#         wemb = torchtext.vocab.GloVe()

#         assert wemb.vectors.shape[1] == word_dim

#         # quick-and-dirty trick to improve word-hit rate
#         missing_words = []
#         for word, idx in word2idx.items():
#             if word not in wemb.stoi:
#                 word = word.replace(
#                     '-', '').replace('.', '').replace("'", '')
#                 if '/' in word:
#                     word = word.split('/')[0]
#             if word in wemb.stoi:
#                 self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
#             else:
#                 missing_words.append(word)
#         print('Words: {}/{} found in vocabulary; {} words missing'.format(
#             len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

#     def forward(self, x, lengths):
#         # Embed word ids to vectors
#         x = self.dropout(self.embed(x))
#         packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

#         # Forward propagate RNN
#         out, _ = self.rnn(packed)

#         # Reshape *final* output to (batch_size, hidden_size)
#         padded = pad_packed_sequence(out, batch_first=True)
#         cap_emb, cap_len = padded

#         if self.use_bidirectional_rnn:
#             cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] +
#                        cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

#         return cap_emb

# # ================================
# # Vision Multi-Scale Fusion Module
# # ================================
# class VMSF(nn.Module):
#     def __init__(self, args):
#         super(VMSF, self).__init__()
#         self.embed_dim = args.embed_dim
#         self.dropout_r = 0.2
#         self.use_relu = True

#         self.conv_512 = nn.Conv2d(in_channels=512, out_channels=self.embed_dim, kernel_size=1, stride=1)
#         self.conv_1024 = nn.Conv2d(in_channels=1024, out_channels=self.embed_dim, kernel_size=1, stride=1)
#         self.conv_2048 = nn.Conv2d(in_channels=2048, out_channels=self.embed_dim * 2, kernel_size=1, stride=1)

#         self.up_sample_double = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_sample_half = nn.Upsample(scale_factor=0.5, mode='nearest')

#         self.channel_att = nn.Sequential(
#             nn.Conv2d(in_channels=self.embed_dim * 4, out_channels=self.embed_dim, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(self.embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True))

#         self.channel_filter = MLP(self.embed_dim, self.embed_dim * 2, self.embed_dim, self.dropout_r, self.use_relu)

#     def forward(self, deep_feas):
#         d1, d2, d3 = deep_feas

#         p_2 = self.conv_1024(d2)
#         up_4 = self.up_sample_double(self.conv_2048(d3))
#         up_2 = self.up_sample_half(self.conv_512(d1))

#         # Depth concat && channel attention
#         ms_fea = self.channel_att(torch.cat([up_2, p_2, up_4], dim=1))

#         # Mean fsuion && chanel filter
#         high_emb = self.channel_filter(ms_fea.mean(-1).mean(-1))

#         return high_emb

# #=================================
# # Scene Fine-Grained Sensing Module
# #=================================
# class SFGS(nn.Module):
#     def __init__(self ,args, dim = 32):
#         super(SFGS,self).__init__()
#         self.args = args
#         self.embed_dim = args.embed_dim
#         self.dim = dim
#         self.dropout_r = 0.1
#         self.use_relu = True

#         self.conv2d_block_11 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((16,16))
#         )
#         self.conv2d_block_33 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=3, bias=False),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((16, 16))
#         )
#         self.conv2d_block_55 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=5, bias=False),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((16, 16))
#         )

#         self.fc = FC(self.embed_dim // 2, self.embed_dim , self.dropout_r, self.use_relu)

#         self.wsa = WSA(args, num_dim=128, is_weighted=True)

#     def forward(self, vl_fea):
#         bs, dim, _, _ = vl_fea.size()

#         vl_1 = self.conv2d_block_11(vl_fea).view(bs, dim, -1)
#         vl_2 = self.conv2d_block_33(vl_fea).view(bs, dim, -1)
#         vl_3 = self.conv2d_block_55(vl_fea).view(bs, dim * 2, -1)

#         vl_depth = torch.cat([vl_1,vl_2,vl_3], dim=1)

#         return self.wsa(self.fc(vl_depth)).mean(1)

# # #=======================================
# # Global-Local embeddings Aggregation Module
# #=======================================
# class Aggregation(nn.Module):
#     def __init__(self, args):
#         super(Aggregation, self).__init__()
#         self.args = args
#         self.embed_dim = args.embed_dim

#         self.fc_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
#         self.fc_2 = nn.Linear(self.embed_dim,self.embed_dim)

#     def forward(self, vl_emb, gl_emb):

#         # Depth concat
#         v_emb = torch.cat([vl_emb,gl_emb],dim=1)

#         return self.fc_2(torch.relu(self.fc_1(v_emb)))

# #========================================
# # Text Coarse-Grained Enhancement Module
# #========================================
# class TCGE(nn.Module):
#     def __init__(self,args):
#         super(TCGE, self).__init__()
#         self.embed_dim = args.embed_dim
#         self.gpuid = args.gpuid

#         self.bn_1d = nn.BatchNorm1d(self.embed_dim)
#         self.ga = GA(args)

#         self.dropout = nn.Dropout(0.2)

#         self.mlp = MLP(self.embed_dim, self.embed_dim * 2, self.embed_dim, 0.1, True)

#         self.conv1d_block_22 = nn.Sequential(
#             nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, stride = 2, kernel_size= 2),
#             nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )
#         self.conv1d_block_33 = nn.Sequential(
#             nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, stride = 3, kernel_size= 3),
#             nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, cap_emb, lengths):
#         # GA Embeddings
#         bs, dim, emb_dim = cap_emb.size()
#         ga_emb = cap_emb + self.dropout(self.bn_1d(self.ga(cap_emb).view(bs * dim, -1)).view(bs, dim, -1))
#         # Joint Wrod Embeddings
#         f2 = self.conv1d_block_22(cap_emb.permute(0, 2, 1)).permute(0, 2, 1)
#         f3 = self.conv1d_block_33(cap_emb.permute(0, 2, 1)).permute(0, 2, 1)
#         jw_emb = torch.cat([f2, f3],dim=1)

#         # GA-JW Fusion
#         ga_jw = torch.cat([ga_emb, jw_emb], dim=1)
#         tex_emb = self.mlp(ga_jw) + ga_jw

#         I = torch.LongTensor(lengths).view(-1, 1, 1) # 100, 1, 1
#         I = Variable(I.expand(tex_emb.size(0), 1, self.embed_dim)-1).cuda(self.gpuid) # 100, 1, 512
#         out = torch.gather(tex_emb, 1, I).squeeze(1)

#         return l2norm(out, dim=-1)

# #======================
# # Multi-Head Attention
# #======================
# class MHAtt(nn.Module):
#     def __init__(self, args):
#         super(MHAtt, self).__init__()
#         self.embed_dim = args.embed_dim
#         self.dropout_r = 0.1
#         self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
#         self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
#         self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
#         self.linear_merge = nn.Linear(self.embed_dim, self.embed_dim)

#         self.dropout = nn.Dropout(self.dropout_r)

#     def forward(self, v, k, q, mask=None):
#         bs = q.size(0)

#         v = self.linear_v(v).view(bs, -1, 8, 64).transpose(1, 2)
#         k = self.linear_k(k).view(bs, -1, 8, 64).transpose(1, 2)
#         q = self.linear_q(q).view(bs, -1, 8, 64).transpose(1, 2)

#         atted = self.att(v, k, q, mask)
#         atted = atted.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)

#         atted = self.linear_merge(atted)

#         return atted

#     def att(self, k, q, v, mask=None):
#         d_k = q.shape[-1]

#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

#         if mask is not None:
#             scores = scores.masked_fill(mask, -1e9)

#         att_map = torch.softmax(scores, dim=-1)
#         att_map = self.dropout(att_map)

#         return torch.matmul(att_map, v)

# #============================
# # Weighted Self Attention
# #============================
# class WSA(nn.Module):
#     def __init__(self,args, num_dim = 128, is_weighted = False):
#         super(WSA, self).__init__()
#         self.num_dim = num_dim
#         self.embed_dim = args.embed_dim
#         self.is_weighted = is_weighted
#         self.dropout_r = 0.1

#         self.mhatt = MHAtt(args)
#         self.ffn = FeedForward(self.embed_dim, self.embed_dim * 2)

#         self.dropout1 = nn.Dropout(self.dropout_r)
#         self.norm1 = nn.LayerNorm(self.embed_dim)

#         self.dropout2 = nn.Dropout(self.dropout_r)
#         self.norm2 = nn.LayerNorm(self.embed_dim)

#         # Learnable weights
#         if is_weighted:
#             self.fmp_weight = nn.Parameter(torch.randn(1, self.num_dim, self.embed_dim))
#     def forward(self, x, x_mask=None):
#         bs = x.shape[0]

#===============
#         x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
#         x = self.norm2(x + self.dropout2(self.ffn(x)))

#         if self.is_weighted:
#             # feature map fusion
#             x = self.fmp_weight.expand(bs, x.shape[1], x.shape[2]).transpose(1, 2).bmm(x)

#         return x

# #===================
# # Gated Attention
# #===================
# class GA(nn.Module):
#     def __init__(self, args):
#         super(GA, self).__init__()
#         self.h = 2
#         self.embed_dim = args.embed_dim
#         self.d_k = self.embed_dim // self.h

#         self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 3)

#         self.fc_q = nn.Linear(self.d_k, self.d_k)
#         self.fc_k = nn.Linear(self.d_k, self.d_k)
#         self.fc_g = nn.Linear(self.d_k, self.d_k*2)

#     def forward(self, cap_emb):
#         bs = cap_emb.shape[0]

#         q, k, v = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
#                    for l, x in zip(self.linears, (cap_emb, cap_emb, cap_emb))]

#         # gate
#         G = self.fc_q(q) * self.fc_k(k)
#         M = torch.sigmoid(self.fc_g(G)) # (bs, h, num_region, d_k*2)
#         q = q * M[:, :, :, :self.d_k]
#         k = k * M[:, :, :, self.d_k:]

#         scores = torch.div(torch.matmul(q, k.transpose(-2, -1)), math.sqrt(self.d_k), rounding_mode='floor')

#         p_attn = torch.softmax(scores, dim=-1)

#         x = torch.matmul(p_attn, v)
#         x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)

#         return x
# ===
# # Some Reuse Module
# #==================
# # full connection layer
# class FC(nn.Module):
#     def __init__(self, in_size, out_size, dropout=0., use_relu=True):
#         super(FC, self).__init__()
#         self.dropout_r = dropout
#         self.use_relu = use_relu
#         self.linear = nn.Linear(in_size, out_size)

#         if use_relu:
#             self.relu = nn.ReLU(inplace=True)

#         if dropout > 0:
#             self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.linear(x)

#         if self.use_relu:
#             x = self.relu(x)

#         if self.dropout_r > 0:
#             x = self.dropout(x)

#         return x


# # Feed Forward Nets
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super(FeedForward,self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

# # mlp
# class MLP(nn.Module):
#     def __init__(self, in_size, mid_size, out_size, dropout=0., use_relu=True):
#         super(MLP, self).__init__()

#         self.fc = FC(in_size, mid_size, dropout=dropout, use_relu=use_relu)
#         self.linear = nn.Linear(mid_size, out_size)

#     def forward(self, x):
#         out = self.linear(self.fc(x))
#         return out

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

def factory(args, 
            # word2idx, 
            cuda=True, 
            data_parallel=False):
    args_new = copy.copy(args)

    # model_without_ddp = SWAN(args_new, word2idx)
    model_without_ddp = UrbanCross(args_new, 
                                #    word2idx
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

def factory_finetune(
                     args, 
                    #  word2idx, 
                     cuda=True, 
                     data_parallel=False
                     ):
    args_new = copy.copy(args)

    # model_without_ddp = SWAN(args_new, word2idx)
    model_without_ddp = UrbanCross_finetune(args_new, 
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

def factory_wo_seg(
                     args, 
                    #  word2idx, 
                     cuda=True, 
                     data_parallel=False
                     ):
    args_new = copy.copy(args)

    # model_without_ddp = SWAN(args_new, word2idx)
    model_without_ddp = UrbanCross_wo_seg(args_new, 
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
