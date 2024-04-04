import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm


def show_anns(anns, ori_img, img_path):
    """
    Display annotations (masks) on the original image and save the segmented images.

    Args:
        anns (list): List of annotations (masks).
        ori_img (numpy.ndarray): Original image.
        img_path (str): Path to the original image.
    """
    img_name = img_path.split("/")[-1]
    seg_path = img_path.replace("/images/", "/image_segments/").split(".")[0]

    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    # anns就是整个masks的list传过来了
    if len(anns) == 0:
        return

    # 按面积从大到小排序
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4,))
    img[:, :, 3] = 0    # img[256,256,4]
    
    for idx, ann in enumerate(sorted_anns):
        m = ann["segmentation"]
        # m是[534,800], 是一个true和false的numpy array
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
  
    ax.imshow(img)
    plt.axis("off")
    plt.savefig(os.path.join(seg_path, img_name), bbox_inches="tight", pad_inches=0)


def show_masks_mine(anns, ori_img, img_path):
    """
    Display masks on the original image and save segmented images.

    Args:
        anns (list): List of annotations (masks).
        ori_img (numpy.ndarray): Original image.
        img_path (str): Path to the original image.
    """
    img_name = img_path.split("/")[-1].split(".")[0]
    seg_path = img_path.replace("/images/", "/image_segments/").split(".")[0]

    # anns就是整个masks的list传过来了
    if len(anns) == 0:
        return
    
    # 按面积从大到小排序
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for idx, ann in enumerate(sorted_anns):
        m = ann["segmentation"]
        masked_img = np.copy(ori_img)
        masked_img[m != 1] = [255, 255, 255]

        plt.clf()
        plt.imshow(masked_img)
        plt.axis("off")

        plt.savefig(
            os.path.join(seg_path, img_name + "_" + str(idx) + ".jpg"),
            bbox_inches="tight",
            pad_inches=0,
        )


if __name__ == "__main__":
    img_path = "/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/images"
    df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/dataset_rsicd.csv")

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    mask_generator = SamAutomaticMaskGenerator(sam)

    img_lists = df["image_name"]
    # for idx, row in tqdm(df.iterrows()):
    # for idx, i in enumerate(tqdm(img_lists[::-1])):
    # for idx, i in enumerate(tqdm(img_lists[::-1][7000:])):
    # for idx, i in enumerate(tqdm(img_lists)):
    # for idx, i in enumerate(tqdm(img_lists[40000:])):
    for idx, i in enumerate(tqdm(img_lists)):
        seg_path = os.path.join(img_path[:-6] + "image_segments/", i.split(".")[0])
        if os.path.exists(seg_path):
            continue
        
        image = cv2.imread(os.path.join(img_path, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(image)

        plt.imshow(image)
        show_anns(masks, image, os.path.join(img_path, i))
        masks = masks[:50]
        show_masks_mine(masks, image, os.path.join(img_path, i))
