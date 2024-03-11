import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas as pd
# import yaml
import argparse
import utils
from vocab import deserialize_vocab
from PIL import Image
import open_clip

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, args, data_split, vocab):
        self.vocab = vocab
        self.loc = args.data_path  #'./data/rsitmd_precomp/'
        self.img_path = args.image_path  #./rs_data/rsitmd/images/
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # Captions
        self.captions = []
        self.maxlength = 0

        # import ipdb;ipdb.set_trace()
        if data_split != 'test':
            #./data/rsitmd_precomp/train_caps_verify.txt
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []

            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                # transforms.RandomCrop(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.transform_segment = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        else:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]

        vocab = self.vocab
        # import ipdb;ipdb.set_trace()
        tokens_clip = self.clip_tokenizer(
                        caption.lower().decode('utf-8')
                    )  # [1, 77]
        
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8')
        )
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]


        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)
        # import ipdb;ipdb.set_trace()
        image = Image.open(self.img_path +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])
        img_name = str(self.images[img_id])[2:-1].split('.')[0]
        seg_path = os.path.join(self.img_path.replace('images', 'images_segment'), img_name)
        num_seg = 10
        seg_list = []
        for i in range(num_seg):
            seg_list.append(
                self.transform_segment(
                    Image.open(os.path.join(seg_path,img_name+f'_{i}'+'.jpg')).convert('RGB')
                )
            )
            
        segment_img = torch.stack(seg_list,dim=0)
            
        # import ipdb;ipdb.set_trace()
        # return image, caption, tokens_UNK, index, img_id, tokens_clip
        return image, caption, tokens_UNK, index, img_id, tokens_clip, segment_img


    def __len__(self):
        return self.length


class PrecompDataset_mine(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, args, data_split, vocab):
        self.country = "Spain"
        
        self.vocab = vocab
        self.loc = args.data_path  #'./data/rsitmd_precomp/'
        self.img_path = os.path.join(args.image_path, self.country, 'images')  #./rs_data/rsitmd/images/
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # Captions
        self.captions = []
        self.maxlength = 0

        # import ipdb;ipdb.set_trace()
        # if data_split != 'test':
        if data_split == 'train' or data_split == 'val':
            df = pd.read_csv(f'urbancross_data/instructblip_generation_with_tag/instructblip_generation_{self.country.lower()}_refine.csv')
            df1 = pd.read_csv(f"urbancross_data/images_target/{self.country}/captions_top30.csv")
            # import ipdb; ipdb.set_trace()
            # #./data/rsitmd_precomp/train_caps_verify.txt
            # with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
            #     for line in f:
            #         self.captions.append(line.strip())

            # self.images = []

            # with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
            #     for line in f:
            #         self.images.append(line.strip())
        else:
            df = pd.read_csv(f'urbancross_data/instructblip_generation_with_tag/instructblip_generation_{self.country.lower()}_refine.csv')
            df1 = pd.read_csv(f"urbancross_data/images_target/{self.country}/captions_top30.csv")
            

            # with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
            #     for line in f:
            #         self.captions.append(line.strip())

            # self.images = []
            # with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
            #     for line in f:
            #         self.images.append(line.strip())

        self.captions = df['description'].values.tolist()
        self.images = df['image_name'].values.tolist()
        self.tags = df1['title_multi_objects'].values.tolist()  
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        # if len(self.images) != self.length:
        #     self.im_div = 5
        # else:
        #     self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                # transforms.RandomCrop(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.transform_segment = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        else:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        # self.im_div  5
        # img_id = index//self.im_div
        img_id = index
        caption = self.captions[index]
        tag = self.tags[index]
        vocab = self.vocab
        # import ipdb;ipdb.set_trace()
        cap_tokens = self.clip_tokenizer(
                        caption
                    )  # [1, 77]
        tag_tokens = self.clip_tokenizer(
                tag
            )  # [1, 77]
        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(
        #     caption.lower().decode('utf-8'))
        # punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        # tokens = [k for k in tokens if k not in punctuations]
        # tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]


        # caption = []
        # caption.extend([vocab(token) for token in tokens_UNK])
        # caption = torch.LongTensor(caption)
        # import ipdb;ipdb.set_trace()
        image = Image.open(
                    os.path.join(self.img_path, self.images[img_id])
                ).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])
        img_name = self.images[img_id].split('.')[0]
        seg_path = os.path.join(self.img_path[:-6] + 'image_segments/', img_name)
        num_seg = 10
        seg_list = []
        for i in range(num_seg):
            # if not os.path.exists(os.path.join(seg_path,img_name+f'_{i}'+'.jpg')):
            #     seg_list.append(
            #         torch.rand(3, 224, 224)
            #     )
            # else:
            seg_list.append(
                self.transform_segment(
                    Image.open(os.path.join(seg_path, img_name+f'_{i}'+'.jpg')).convert('RGB')
                )
            )
            
        # import ipdb;ipdb.set_trace()
        segment_img = torch.stack(seg_list,dim=0)
            
        # import ipdb;ipdb.set_trace()
        # return image, caption, tokens_UNK, index, img_id, tokens_clip
        return image, caption, tag, index, img_id, cap_tokens, tag_tokens, segment_img


    def __len__(self):
        return self.length

def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, captions, tokens, ids, img_ids, tokens_clip, segment_img = zip(*data)
    # return image, caption, tokens_UNK, index, img_id, tokens_clip, segment_img
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    segment_img = torch.stack(segment_img, 0)
    # import ipdb;ipdb.set_trace()
    tokens_clip = torch.cat(tokens_clip, dim=0)

    import ipdb; ipdb.set_trace()
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    import ipdb; ipdb.set_trace()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, targets, lengths, ids, tokens_clip, segment_img

def collate_fn_mine(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, captions, tags, ids, img_ids, cap_tokens, tag_tokens, segment_img = zip(*data)

    
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    segment_img = torch.stack(segment_img, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)
    tag_tokens = torch.cat(tag_tokens, dim=0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    # import ipdb;ipdb.set_trace()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]

    # lengths = [l if l !=0 else 1 for l in lengths]

    # return images, targets, lengths, ids, cap_tokens, segment_img, tag_tokens
    return images, ids, cap_tokens, segment_img, tag_tokens
    

def get_precomp_loader(args, 
                       data_split, 
                       vocab, 
                       batch_size=100,
                       shuffle=False, 
                       num_workers=0
                       ):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(args, data_split, vocab)
    if args.distributed and data_split == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                #   pin_memory=False,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                #   pin_memory=False,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers)
    return data_loader

def get_precomp_loader_mine(
                        args, 
                        data_split, 
                        vocab, 
                        batch_size=100,
                        shuffle=False, 
                        num_workers=0
                       ):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset_mine(args, data_split, vocab)
    if args.distributed and data_split == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        data_loader = torch.utils.data.DataLoader(
                    dataset=dset,
                    batch_size=batch_size,
                    pin_memory=True,
                    #pin_memory=False,
                    collate_fn=collate_fn_mine,
                    num_workers=num_workers,
                    sampler=sampler
        )
    else:
        data_loader = torch.utils.data.DataLoader(
                            dataset=dset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True,
                            #   pin_memory=False,
                            collate_fn=collate_fn_mine,
                            num_workers=num_workers
        )
    return data_loader

def get_loaders(args, vocab):
    train_loader = get_precomp_loader(args, 
                                      data_split = 'train', 
                                      vocab = vocab,
                                      batch_size = args.batch_size, 
                                      shuffle = True, 
                                      num_workers = args.workers
                                      )
    
    val_loader = get_precomp_loader(args, 
                                    'val',
                                    vocab,
                                    args.batch_size_val, 
                                    False, 
                                    args.workers
                                    )
    return train_loader, val_loader

def get_loaders_mine(args, vocab):
    train_loader = get_precomp_loader_mine(args, 
                                      data_split = 'train', 
                                      vocab = vocab,
                                      batch_size = args.batch_size, 
                                      shuffle = True, 
                                      num_workers = args.workers
                                      )
    val_loader = get_precomp_loader_mine(args, 
                                        'val',
                                        vocab,
                                        args.batch_size_val, 
                                        False, 
                                        args.workers
                                    )
    return train_loader, val_loader
def get_test_loader(args, vocab):
    test_loader = get_precomp_loader(args, 'test', vocab,
                                      args.batch_size_val, False, args.workers)
    return test_loader
