import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas as pd
import argparse
import utils
from utils.vocab import deserialize_vocab
from PIL import Image
import open_clip
from tqdm import tqdm


class PrecompDataset(data.Dataset):
    """
    Dataset class for loading precomputed captions and image features.
    """

    def __init__(self, args, data_split, vocab):
        """
        Initialize the PrecompDataset.

        Args:
            args (argparse.Namespace): Parsed arguments.
            data_split (str): Data split ("train", "val", or "test").
            vocab (Vocabulary): Vocabulary object.
        """
        self.vocab = vocab
        self.loc = args.data_path
        self.img_path = args.image_path  # ./rs_data/rsitmd/images/
        self.clip_tokenizer = open_clip.get_tokenizer(
            "ViT-L-14"
        )  # Use CLIP's tokenizer

        # Load captions and image filenames
        if data_split != "test":
            captions_file = f"{data_split}_caps_verify.txt"
            filename_file = f"{data_split}_filename_verify.txt"
        else:
            captions_file = f"{data_split}_caps.txt"
            filename_file = f"{data_split}_filename.txt"

        with open(os.path.join(self.loc, captions_file), "r") as f:
            self.captions = [line.strip() for line in f]

        with open(os.path.join(self.loc, filename_file), "r") as f:
            self.images = [line.strip() for line in f]

        self.length = len(self.captions)
        self.im_div = 5 if len(self.images) != self.length else 1

        # Define transformations based on data split
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

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image, caption, tokens, and other metadata.
        """
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]

        # Tokenize caption using CLIP tokenizer
        vocab = self.vocab
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
                self.img_path = os.path.join(args.image_path, args.country, "images")
            else:
                if finetune == "source":
                    args.country = args.source_country
                    self.img_path = os.path.join(args.image_path, args.country, "images")
                else:
                    args.country = args.target_country
                    self.img_path = os.path.join(args.image_path, args.country, "images")
            
            df = pd.read_csv(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{args.country}/instructblip_generation_{args.country.lower()}_refine.csv")
            data_split_txt = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{args.country}/{data_split}_list.txt"
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
        image = Image.open(os.path.join(self.img_path, self.images[img_id])).convert(
            "RGB"
        )
        # Apply transformations to the image, including resizing, random rotation, random cropping, and normalization
        image = self.transform(image)

        # Construct the path for image segments
        img_name = self.images[img_id].split(".")[0]
        seg_path = os.path.join(self.img_path[:-6] + "image_segments/", img_name)

        # Get the current number of image segments
        current_num_seg = min(len(os.listdir(seg_path)) - 1, self.num_seg)

        seg_list = []
        # Iterate over each image segment, load and apply transformations
        img_list = os.listdir(os.path.join(seg_path))
        for i in range(current_num_seg):
            seg_list.append(
                self.transform_segment(
                    Image.open(os.path.join(seg_path + "/" + img_list[i])).convert("RGB")
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
    

class PrecompDataset_without_sam_mine(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, args, data_split, finetune=None):
        if args.country:
            # If country is not specified, set image path based on finetune and country arguments
            if finetune is None:
                self.img_path = os.path.join(args.image_path, args.country, "images")
            else:
                if finetune == "source":
                    args.country = args.source_country
                    self.img_path = os.path.join(
                        args.image_path, args.country, "images"
                    )
                else:
                    args.country = args.target_country
                    self.img_path = os.path.join(
                        args.image_path, args.country, "images"
                    )

            df = pd.read_csv(
                f"{args.image_path}/{args.country}/instructblip_generation_{args.country.lower()}_refine.csv"
            )
            data_split_txt = f"{args.image_path}/{args.country}/{data_split}_list.txt"
        else:
            # If country is not specified, set image path and data split text file path based on the data split
            if args.data_name == "rsicd":
                self.img_path = args.image_path
                df = pd.read_csv(
                    "/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/dataset_rsicd.csv"
                )
                data_split_txt = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/{data_split}_list.txt"
            elif args.data_name == "rsitmd":
                self.img_path = args.image_path
                df = pd.read_csv(
                    "/hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/dataset_rsitmd.csv"
                )
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
        image = Image.open(os.path.join(self.img_path, self.images[img_id])).convert(
            "RGB"
        )
        # Apply transformations to the image, including resizing, random rotation, random cropping, and normalization
        image = self.transform(image)

        # Return the image, description, index, image ID, caption token sequence, and image segment tensor
        return image, caption, index, img_id, cap_tokens

    def __len__(self):
        return self.length


class PrecompDataset_mine_finetune(data.Dataset):
    """
    Load precomputed captions and image features for fine-tuning.
    """

    def __init__(self, args, data_split, country, source=True):
        self.img_path = os.path.join(args.image_path, country, "images")
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.captions = []

        df = pd.read_csv(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/instructblip_generation_{country.lower()}_refine.csv")
        split_list = []

        if source:
            path_ = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/{data_split}_list.txt"
        else:
            if data_split == "train":
                path_ = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/finetune_list.txt"
            else:
                path_ = f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/finetune_val_list.txt"

        with open(path_, "r") as f:
            for line in f:
                split_list.append(line.strip())

        df = df[df["image_name"].isin(split_list)]
        self.captions = df["description"].values.tolist()
        self.images = df["image_name"].values.tolist()
        self.length = len(self.captions)

        if data_split == "train":
            self.transform = transforms.Compose([
                    transforms.Resize((278, 278)),
                    transforms.RandomRotation(degrees=(0, 90)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.transform_segment = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.transform_segment = self.transform

    def __getitem__(self, index):
        img_id = index
        caption = self.captions[index]
        cap_tokens = self.clip_tokenizer(caption)  # [1, 77]

        image = Image.open(os.path.join(self.img_path, self.images[img_id])).convert("RGB")
        image = self.transform(image)  # torch.Size([3, 256, 256])

        return image, caption, index, img_id, cap_tokens

    def __len__(self):
        return self.length


class PrecompDataset_mine_zeroshot(data.Dataset):
    """
    Load precomputed captions and image features
    """
    def __init__(self, args, data_split, country):
        self.img_path = os.path.join(args.image_path, country, 'images')
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.captions = []

        df = pd.read_csv(f'/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/instructblip_generation_with_tag/instructblip_generation_{country.lower()}_refine.csv')
        split_list = []
        path_ = f'/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/zeroshot_list.txt'
        
        with open(path_, 'r') as f:
            for line in tqdm(f):
                # Remove newlines at the end of lines and add to list
                split_list.append(line.strip())
        df = df[df['image_name'].isin(split_list)]
        self.captions = df['description'].values.tolist()
        self.images = df['image_name'].values.tolist()
        self.length = len(self.captions)

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.transform_segment = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.transform_segment = self.transform
            
    def __getitem__(self, index):
        img_id = index
        caption = self.captions[index]

        cap_tokens = self.clip_tokenizer(
                        caption
                    )  # [1, 77]
        
        image = Image.open(
                    os.path.join(self.img_path, self.images[img_id])
                ).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])
        
        return image, caption, index, img_id, cap_tokens, os.path.join(self.img_path, self.images[img_id])


    def __len__(self):
        return self.length
 

def collate_fn(data):
    """
    Custom collate function to be used with DataLoader for handling variable length captions.

    Args:
        data (list): List of tuples (image, caption, tokens, index, img_id, tokens_clip, segment_img).

    Returns:
        torch.Tensor: Stacked images tensor.
        torch.Tensor: Padded and stacked captions tensor.
        list: Lengths of each caption.
        list: Indices.
        torch.Tensor: Concatenated tokens_clip tensor.
        torch.Tensor: Stacked segment images tensor.
    """
    # Sort the data list by caption length in descending order
    data.sort(key=lambda x: len(x[2]), reverse=True)
    # Unpack the data tuples
    images, captions, tokens, ids, img_ids, tokens_clip, segment_img = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    segment_img = torch.stack(segment_img, 0)
    tokens_clip = torch.cat(tokens_clip, dim=0)

    # Merge captions (pad and convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # Ensure that lengths are at least 1 to avoid division by zero
    lengths = [l if l != 0 else 1 for l in lengths]

    return images, targets, lengths, ids, tokens_clip, segment_img


def collate_fn_mine(data):
    # Unpack the data tuples
    images, captions, ids, img_ids, cap_tokens, segment_img = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    segment_img = torch.stack(segment_img, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)

    return images, ids, cap_tokens, segment_img


def collate_fn_without_sam_mine(data):
    # Unpack the data tuples
    images, captions, ids, img_ids, cap_tokens = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)

    return images, ids, cap_tokens


def collate_fn_mine_finetune(data):
    """
    Custom collate function for fine-tuning your specific dataset structure.

    Args:
        data (list): List of tuples (image, caption, ids, img_ids, cap_tokens).

    Returns:
        torch.Tensor: Stacked images tensor.
        list: Indices.
        torch.Tensor: Concatenated caption tokens tensor.
    """
    # Unpack the data tuples
    images, caption, ids, img_ids, cap_tokens = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)

    # Return the necessary components
    return images, cap_tokens


def collate_fn_mine_zeroshot(data):
    
    images, caption, ids, img_ids, cap_tokens, img_path = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    cap_tokens = torch.cat(cap_tokens, dim=0)
    img_path = list(img_path)
    caption = list(caption)
    
    return images, cap_tokens, img_path, caption


def get_precomp_loader(args, data_split, vocab, batch_size=100, shuffle=False, num_workers=0):
    """
    Returns torch.utils.data.DataLoader for custom precomputed dataset.

    Args:
        args (argparse.Namespace): Parsed arguments.
        data_split (str): Dataset split ('train', 'val', 'test').
        vocab (Vocabulary): Vocabulary object.
        batch_size (int, optional): Batch size. Defaults to 100.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the custom precomputed dataset.
    """
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


def get_precomp_loader_without_sam_mine(
    args,
    data_split,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    finetune=None,
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset_without_sam_mine(
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
            collate_fn=collate_fn_without_sam_mine,
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
            collate_fn=collate_fn_without_sam_mine,
            num_workers=num_workers,
            drop_last=True,
        )
    return data_loader


def get_precomp_loader_mine_finetune(
    args,
    data_split,
    country,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    source=True,
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset_mine_finetune(
        args,
        data_split,
        country=country,
        source=source,
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
    return data_loader, dset


def get_precomp_loader_mine_zeroshot(args, data_split, country, batch_size=100, shuffle=False, num_workers=0, source = True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset_mine_finetune(args, data_split, country=country, source=source)
    if args.distributed and data_split == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=collate_fn_mine_finetune,
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
            collate_fn=collate_fn_mine_finetune,
            num_workers=num_workers,
            drop_last=True,
        )
    return data_loader, dset


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


def get_loaders_without_sam_mine(args):
    train_loader = get_precomp_loader_without_sam_mine(
        args,
        data_split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = get_precomp_loader_without_sam_mine(
        args,
        "val",
        args.batch_size_val,
        False,
        args.workers,
    )
    return train_loader, val_loader


def get_loaders_finetune_backup(args):
    source_train_loader, source_train_dataset = get_precomp_loader_mine_finetune(
        args,
        data_split="train",
        country=args.country_source,
        batch_size=args.batch_size_source,
        shuffle=True,
        num_workers=args.workers,
        source=True,
    )

    target_train_loader, target_train_dataset = get_precomp_loader_mine_finetune(
        args,
        data_split="train",
        country=args.country_target,
        batch_size=args.batch_size_target,
        shuffle=True,
        num_workers=args.workers,
        source=False,
    )

    val_loader_target, val_dataset_target = get_precomp_loader_mine_finetune(
        args,
        "val",
        args.country_target,
        args.batch_size_val_target,
        False,
        args.workers,
        source=False,
    )

    return (
        source_train_loader,
        target_train_loader,
        source_train_dataset,
        target_train_dataset,
        val_loader_target,
        val_dataset_target,
    )


def get_loaders_finetune(args):
    source_train_dataset = PrecompDataset_mine_finetune(
        args,
        data_split="train",
        country=args.country_source,

        source=True,
    )
    source_train_loader = torch.utils.data.DataLoader(
        dataset=source_train_dataset,
        batch_size=args.batch_size_source,
        shuffle=True,
        pin_memory=True,
        #   pin_memory=False,
        collate_fn=collate_fn_mine_finetune,
        num_workers=args.workers,
        drop_last=True,
    )
   
    target_train_dataset = PrecompDataset_mine_finetune(
        args,
        data_split="train",
        country=args.country_target,
        # vocab,
        # finetune,
        source=False,
    )
    target_train_loader = torch.utils.data.DataLoader(
        dataset=target_train_dataset,
        batch_size=args.batch_size_target,
        shuffle=True,
        pin_memory=True,
        #   pin_memory=False,
        collate_fn=collate_fn_mine_finetune,
        num_workers=args.workers,
        drop_last=True,
    )

    val_loader_target, val_dataset_target = get_precomp_loader_mine_finetune(
        args,
        "val",
        args.country_target,
        args.batch_size_val_target,
        False,
        args.workers,
        source=False,
    )

    return (
        source_train_loader,
        target_train_loader,
        source_train_dataset,
        target_train_dataset,
        val_loader_target,
        val_dataset_target,
    )


def get_test_loader(args, vocab):
    test_loader = get_precomp_loader(
        args, "test", vocab, args.batch_size_val, False, args.workers
    )
    return test_loader

def get_test_loader_finetune(args):
    test_loader = get_precomp_loader_mine_finetune(
                                    args, 
                                    'test', 
                                    args.batch_size_val, False, args.workers)
    return test_loader

def get_test_loader_mine(args):
    test_loader = get_precomp_loader_mine(
        args,
        data_split="test",
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
    )

    return test_loader


def get_test_loader_without_sam_mine(args):
    test_loader = get_precomp_loader_without_sam_mine(
        args,
        data_split="test",
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
    )

    return test_loader


def get_test_loader_zeroshot(args):
    dset = PrecompDataset_mine_zeroshot(args, 'test', country=args.country)
    test_loader = torch.utils.data.DataLoader(
                        dataset=dset,
                        batch_size=args.batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        collate_fn=collate_fn_mine_zeroshot,
                        num_workers=args.workers,
                        drop_last=True,
    )
    
    return test_loader, dset