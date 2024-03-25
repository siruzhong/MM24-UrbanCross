import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas as pd
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
        self.img_path = args.image_path  # ./rs_data/rsitmd/images/
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.captions = []
        self.maxlength = 0

        if data_split != "test":
            # ./data/rsitmd_precomp/train_caps_verify.txt
            with open(self.loc + "%s_caps_verify.txt" % data_split, "rb") as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []

            with open(self.loc + "%s_filename_verify.txt" % data_split, "rb") as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + "%s_caps.txt" % data_split, "rb") as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + "%s_filename.txt" % data_split, "rb") as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((278, 278)),
                    transforms.RandomRotation(degrees=(0, 90)),
                    # transforms.RandomCrop(256),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = transforms.Compose(
                [
                    # transforms.Resize((256, 256)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((256, 256)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]

        vocab = self.vocab
        # import ipdb;ipdb.set_trace()
        tokens_clip = self.clip_tokenizer(caption.lower().decode("utf-8"))  # [1, 77]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower().decode("utf-8"))
        punctuations = [
            ",",
            ".",
            ":",
            ";",
            "?",
            "(",
            ")",
            "[",
            "]",
            "&",
            "!",
            "*",
            "@",
            "#",
            "$",
            "%",
        ]
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else "<unk>" for k in tokens]

        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)
        # import ipdb;ipdb.set_trace()
        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert(
            "RGB"
        )
        image = self.transform(image)  # torch.Size([3, 256, 256])
        img_name = str(self.images[img_id])[2:-1].split(".")[0]
        seg_path = os.path.join(
            self.img_path.replace("images", "images_segment"), img_name
        )
        num_seg = 10
        seg_list = []
        for i in range(num_seg):
            seg_list.append(
                self.transform_segment(
                    Image.open(
                        os.path.join(seg_path, img_name + f"_{i}" + ".jpg")
                    ).convert("RGB")
                )
            )

        segment_img = torch.stack(seg_list, dim=0)

        # import ipdb;ipdb.set_trace()
        # return image, caption, tokens_UNK, index, img_id, tokens_clip
        return image, caption, tokens_UNK, index, img_id, tokens_clip, segment_img

    def __len__(self):
        return self.length


class PrecompDataset_mine(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, args, data_split, finetune=None):
        if args.country:
            # If country is not specified, set image path based on finetune and country arguments
            if finetune is None:
                self.img_path = os.path.join(
                    args.image_path, args.country, "images"
                )
            else:
                if finetune == "source":
                    args.country = args.source_country
                    self.img_path = os.path.join(args.image_path, args.country, "images")
                else:
                    args.country = args.target_country
                    self.img_path = os.path.join(args.image_path, args.country, "images")
            
            df = pd.read_csv(f"urbancross_data/instructblip_generation_with_tag/instructblip_generation_{args.country.lower()}_refine.csv")
            data_split_txt = f"urbancross_data/images_target/{args.country}/{data_split}_list.txt"
        else:
            # If country is not specified, set image path and data split text file path based on the data split
            if args.data_name == "rsicd":
                self.img_path = args.image_path
                df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/dataset_rsicd.csv")
                data_split_txt = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/{data_split}_list.txt"
            elif args.data_name == "rsitmd":
                self.img_path = args.image_path        
                df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/dataset_rsitmd.csv")
                data_split_txt = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/{data_split}_list.txt"

        # Initialize OpenAI's CLIP tokenizer
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

        split_list = []
        # Open and read the content of the data split text file into a list
        with open(data_split_txt, "r") as f:
            for line in f:
                # Remove the newline character at the end of each line and add it to the list
                split_list.append(line.strip())

        # Filter the DataFrame to keep only the rows corresponding to the images in the split list
        df = df[df["image_name"].isin(split_list)]

        # Extract captions and image names from the filtered DataFrame
        self.captions = df["description"].values.tolist()
        self.images = df["image_name"].values.tolist()
        self.length = len(self.captions)
        self.num_seg = args.num_seg

        # Define image transformations based on the data split
        if data_split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((278, 278)),
                    transforms.RandomRotation(degrees=(0, 90)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = self.transform

    def __getitem__(self, index):
        # Handle image redundancy
        img_id = index
        # Get the description corresponding to the image
        caption = self.captions[index]
        # Tokenize the description to get the token sequence
        cap_tokens = self.clip_tokenizer(caption)  # [1, 77]
       
        # Load the image
        image = Image.open(os.path.join(self.img_path, self.images[img_id])).convert("RGB")
        # Apply transformations to the image, including resizing, random rotation, random cropping, and normalization
        image = self.transform(image) 
        
        # Construct the path for image segments
        img_name = self.images[img_id].split(".")[0]
        seg_path = os.path.join(self.img_path[:-6] + "image_segments/", img_name)

        # Get the current number of image segments
        current_num_seg = min(len(os.listdir(seg_path)) - 1, self.num_seg)

        seg_list = []
        # Iterate over each image segment, load and apply transformations
        for i in range(current_num_seg):
            seg_list.append(
                self.transform_segment(
                    Image.open(os.path.join(seg_path, img_name + f"_{i}" + ".jpg")).convert("RGB")
                )
            )
        # If the current number of image segments is less than the specified number, fill with zero tensors
        if current_num_seg < self.num_seg:
            for i in range(current_num_seg, self.num_seg):
                seg_list.append(torch.zeros(3, 224, 224))
        
        # Stack the processed image segment tensors
        segment_img = torch.stack(seg_list, dim=0)

        # Return the image, description, index, image ID, caption token sequence, and image segment tensor
        return image, caption, index, img_id, cap_tokens, segment_img

    def __len__(self):
        return self.length


class PrecompDataset_mine_finetune_old(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, args, data_split, vocab, finetune=None):

        self.img_path_source = os.path.join(
            args.image_path, args.country_source, "images"
        )
        # else:
        #     args.country = args.target_country
        self.img_path_target = os.path.join(
            args.image_path, args.country_target, "images"
        )

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # Captions
        self.captions = []
        # self.maxlength = 0

        df_source = pd.read_csv(
            f"urbancross_data/instructblip_generation_with_tag/instructblip_generation_{args.country_source.lower()}_refine.csv"
        )
        # if data_split == 'train' or data_split == 'val':
        split_list = []
        # 打开文件并读取内容到列表中
        with open(
            f"urbancross_data/images_target/{args.country_source}/{data_split}_list.txt",
            "r",
        ) as f:
            for line in f:
                # 去除行末的换行符并添加到列表中
                split_list.append(line.strip())

        df_source = df_source[df_source["image_name"].isin(split_list)]
        self.captions_source = df_source["description"].values.tolist()
        self.images_source = df_source["image_name"].values.tolist()
        self.length = len(self.captions_source)

        df_target = pd.read_csv(
            f"urbancross_data/instructblip_generation_with_tag/instructblip_generation_{args.country_target.lower()}_refine.csv"
        )
        # if data_split == 'train' or data_split == 'val':
        split_list = []
        # 打开文件并读取内容到列表中
        with open(
            f"urbancross_data/images_target/{args.country_target}/{data_split}_list.txt",
            "r",
        ) as f:
            for line in f:
                # 去除行末的换行符并添加到列表中
                split_list.append(line.strip())

        df_target = df_target[df_target["image_name"].isin(split_list)]
        self.captions_target = df_target["description"].values.tolist()
        self.images_target = df_target["image_name"].values.tolist()

        if data_split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((278, 278)),
                    transforms.RandomRotation(degrees=(0, 90)),
                    # transforms.RandomCrop(256),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = transforms.Compose(
                [
                    # transforms.Resize((256, 256)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((256, 256)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = self.transform

    def __getitem__(self, index):
        img_id = index
        caption_source = self.captions_source[index]
        caption_target = self.captions_target[index]

        cap_tokens_source = self.clip_tokenizer(caption_source)  # [1, 77]
        cap_tokens_target = self.clip_tokenizer(caption_target)  # [1, 77]

        image_source = Image.open(
            os.path.join(self.img_path_source, self.images_source[img_id])
        ).convert("RGB")
        image_source = self.transform(image_source)  # torch.Size([3, 256, 256])
        image_target = Image.open(
            os.path.join(self.img_path_target, self.images_target[img_id])
        ).convert("RGB")
        image_target = self.transform(image_target)  # torch.Size([3, 256, 256])

        return (
            image_source,
            image_target,
            caption_source,
            caption_target,
            index,
            img_id,
            cap_tokens_source,
            cap_tokens_target,
        )

    def __len__(self):
        return self.length


class PrecompDataset_mine_finetune(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(
        self,
        args,
        data_split,
        country,
        #  vocab,
        #  finetune=None
    ):

        # import ipdb; ipdb.set_trace()
        self.img_path = os.path.join(args.image_path, country, "images")
        # else:
        #     args.country = args.target_country
        # self.img_path_target = os.path.join(args.image_path, args.country_target, 'images')

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # Captions
        self.captions = []
        # self.maxlength = 0

        df = pd.read_csv(
            f"urbancross_data/instructblip_generation_with_tag/instructblip_generation_{country.lower()}_refine.csv"
        )
        # if data_split == 'train' or data_split == 'val':
        split_list = []
        # 打开文件并读取内容到列表中
        with open(
            f"urbancross_data/images_target/{country}/{data_split}_list.txt", "r"
        ) as f:
            for line in f:
                # 去除行末的换行符并添加到列表中
                split_list.append(line.strip())

        df = df[df["image_name"].isin(split_list)]
        self.captions = df["description"].values.tolist()
        self.images = df["image_name"].values.tolist()
        self.length = len(self.captions)

        # df_target = pd.read_csv(f'urbancross_data/instructblip_generation_with_tag/instructblip_generation_{args.country_target.lower()}_refine.csv')
        # # if data_split == 'train' or data_split == 'val':
        # split_list = []
        # # 打开文件并读取内容到列表中
        # with open(f'urbancross_data/images_target/{args.country_target}/{data_split}_list.txt', 'r') as f:
        #     for line in f:
        #         # 去除行末的换行符并添加到列表中
        #         split_list.append(line.strip())

        # df_target = df_target[df_target['image_name'].isin(split_list)]
        # self.captions_target = df_target['description'].values.tolist()
        # self.images_target = df_target['image_name'].values.tolist()

        if data_split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((278, 278)),
                    transforms.RandomRotation(degrees=(0, 90)),
                    # transforms.RandomCrop(256),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = transforms.Compose(
                [
                    # transforms.Resize((256, 256)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((256, 256)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.transform_segment = self.transform

    def __getitem__(self, index):
        img_id = index
        caption = self.captions[index]
        # caption_target = self.captions_target[index]

        cap_tokens = self.clip_tokenizer(caption)  # [1, 77]
        # cap_tokens_target = self.clip_tokenizer(
        #                 caption_target
        #             )  # [1, 77]

        image = Image.open(os.path.join(self.img_path, self.images[img_id])).convert(
            "RGB"
        )
        image = self.transform(image)  # torch.Size([3, 256, 256])
        # image_target = Image.open(
        #             os.path.join(self.img_path_target, self.images_target[img_id])
        #         ).convert('RGB')
        # image_target = self.transform(image_target)  # torch.Size([3, 256, 256])

        return image, caption, index, img_id, cap_tokens

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

    import ipdb

    ipdb.set_trace()
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    import ipdb

    ipdb.set_trace()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l != 0 else 1 for l in lengths]

    return images, targets, lengths, ids, tokens_clip, segment_img


def collate_fn_mine(data):
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    # images, captions, tags, ids, img_ids, cap_tokens, tag_tokens, segment_img = zip(*data)
    images, captions, ids, img_ids, cap_tokens, segment_img = zip(*data)
    # import ipdb; ipdb.set_trace()

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    segment_img = torch.stack(segment_img, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)
    # tag_tokens = torch.cat(tag_tokens, dim=0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    # import ipdb;ipdb.set_trace()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]

    # lengths = [l if l !=0 else 1 for l in lengths]

    # return images, targets, lengths, ids, cap_tokens, segment_img, tag_tokens
    # return images, ids, cap_tokens, segment_img, tag_tokens
    return images, ids, cap_tokens, segment_img


def collate_fn_mine_finetune_old(data):
    # import ipdb; ipdb.set_trace()
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    # images, captions, tags, ids, img_ids, cap_tokens, tag_tokens, segment_img = zip(*data)
    # images, captions, ids, img_ids, cap_tokens, segment_img = zip(*data)
    (
        images_source,
        images_target,
        caption_source,
        caption_target,
        ids,
        img_ids,
        cap_tokens_source,
        cap_tokens_target,
    ) = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images_source = torch.stack(images_source, 0)
    images_target = torch.stack(images_target, 0)

    # segment_img = torch.stack(segment_img, 0)
    cap_tokens_source = torch.cat(cap_tokens_source, dim=0)
    cap_tokens_target = torch.cat(cap_tokens_target, dim=0)

    # tag_tokens = torch.cat(tag_tokens, dim=0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    # import ipdb;ipdb.set_trace()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]

    # lengths = [l if l !=0 else 1 for l in lengths]

    # return images, targets, lengths, ids, cap_tokens, segment_img, tag_tokens
    # return images, ids, cap_tokens, segment_img, tag_tokens
    return images_source, images_target, ids, cap_tokens_source, cap_tokens_target


def collate_fn_mine_finetune(data):
    # import ipdb; ipdb.set_trace()
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    # images, captions, tags, ids, img_ids, cap_tokens, tag_tokens, segment_img = zip(*data)
    # images, captions, ids, img_ids, cap_tokens, segment_img = zip(*data)
    images, caption, ids, img_ids, cap_tokens = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # images_target = torch.stack(images_target, 0)

    # segment_img = torch.stack(segment_img, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)
    # cap_tokens_target = torch.cat(cap_tokens_target, dim=0)

    # tag_tokens = torch.cat(tag_tokens, dim=0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    # import ipdb;ipdb.set_trace()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]

    # lengths = [l if l !=0 else 1 for l in lengths]

    # return images, targets, lengths, ids, cap_tokens, segment_img, tag_tokens
    # return images, ids, cap_tokens, segment_img, tag_tokens
    return images, ids, cap_tokens


def get_precomp_loader(
    args, data_split, vocab, batch_size=100, shuffle=False, num_workers=0
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(args, data_split, vocab)
    if args.distributed and data_split == "train":
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            pin_memory=True,
            #   pin_memory=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            sampler=sampler,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            #   pin_memory=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    return data_loader


def get_precomp_loader_mine(
    args,
    data_split,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    finetune=None,
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset_mine(
        args,
        data_split,
        finetune,
    )
    if args.distributed and data_split == "train":
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=collate_fn_mine,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate_fn_mine,
            num_workers=num_workers,
            drop_last=True,
        )
    return data_loader


def get_precomp_loader_mine_finetune(
    args,
    data_split,
    # vocab,
    country,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    # finetune=None,
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset_mine_finetune(
        args,
        data_split,
        country=country,
        # vocab,
        # finetune,
    )
    if args.distributed and data_split == "train":
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            pin_memory=True,
            # pin_memory=False,
            collate_fn=collate_fn_mine_finetune,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )
    else:  # this way
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            #   pin_memory=False,
            collate_fn=collate_fn_mine_finetune,
            num_workers=num_workers,
            drop_last=True,
        )
    return data_loader


def get_loaders(args, vocab):
    train_loader = get_precomp_loader(
        args,
        data_split="train",
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = get_precomp_loader(
        args, "val", vocab, args.batch_size_val, False, args.workers
    )
    return train_loader, val_loader


def get_loaders_mine(args):
    train_loader = get_precomp_loader_mine(
        args,
        data_split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    # args.batch_size_val = args.batch_size
    val_loader = get_precomp_loader_mine(
        args,
        "val",
        args.batch_size_val,
        False,
        args.workers,
    )
    return train_loader, val_loader


def get_loaders_finetune(args, vocab):
    source_train_loader = get_precomp_loader_mine_finetune(
        args,
        data_split="train",
        # vocab = vocab,
        country=args.country_source,
        batch_size=args.batch_size_source,
        shuffle=True,
        num_workers=args.workers,
    )

    target_train_loader = get_precomp_loader_mine_finetune(
        args,
        data_split="train",
        # vocab = vocab,
        country=args.country_target,
        batch_size=args.batch_size_target,
        shuffle=True,
        num_workers=args.workers,
    )
    # import ipdb; ipdb.set_trace()
    # args.batch_size_val = args.batch_size
    val_loader_source = get_precomp_loader_mine_finetune(
        args,
        "val",
        # vocab,
        args.country_source,
        args.batch_size_val_source,
        False,
        args.workers,
    )
    val_loader_target = get_precomp_loader_mine_finetune(
        args,
        "val",
        # vocab,
        args.country_target,
        args.batch_size_val_target,
        False,
        args.workers,
    )

    return (
        source_train_loader,
        target_train_loader,
        val_loader_source,
        val_loader_target,
    )


def get_test_loader(args, vocab):
    test_loader = get_precomp_loader(
        args, "test", vocab, args.batch_size_val, False, args.workers
    )
    return test_loader


def get_test_loader_mine(args):
    test_loader = get_precomp_loader_mine(
        args,
        data_split="test",
        #  vocab,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
    )

    return test_loader
