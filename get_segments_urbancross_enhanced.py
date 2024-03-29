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
from transformers import CLIPProcessor, CLIPModel


# 假设环境中已经安装了transformers库并下载了相应的CLIP模型
clip_model = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(clip_model)
processor = CLIPProcessor.from_pretrained(clip_model)


def calculate_clip_similarity(image, text_description):
    """
    计算图像和文本描述之间的相似度。
    """
    image_inputs = processor(images=image, return_tensors="pt", padding=True)
    text_inputs = processor(text=[text_description], return_tensors="pt", padding=True)

    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)

    # 使用余弦相似度作为相似度度量
    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()


# 修改后的函数，用于筛选和保存分割片段
def filter_and_save_segments(
    anns, ori_img, img_path, text_description, area_threshold=1000, topk=5
):
    img_name = img_path.split("/")[-1].split(".")[0]
    seg_path = img_path.replace("/images/", "/enhanced_image_segments/").split(".")[0]

    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    # 筛选面积大于阈值的分割区域
    filtered_anns = [ann for ann in anns if ann["area"] > area_threshold]
    similarities = []

    for ann in filtered_anns:
        mask = ann["segmentation"] == 1
        masked_img = np.copy(ori_img)
        masked_img[~mask] = [255, 255, 255]  # Assuming mask is a boolean array
        sim = calculate_clip_similarity(masked_img, text_description)
        similarities.append(sim)

    # 保留相似度最高的topk个分割区域
    topk_indices = np.argsort(similarities)[-topk:]

    for idx in topk_indices:
        ann = filtered_anns[idx]
        mask = ann["segmentation"] == 1
        masked_img = np.copy(ori_img)
        masked_img[~mask] = [255, 255, 255]

        plt.imshow(masked_img)
        plt.axis("off")
        plt.savefig(
            os.path.join(seg_path, f"{img_name}_{idx}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


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

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0  # img[256,256,4]

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
    df = pd.read_csv(
        "/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/dataset_rsicd.csv"
    )

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    mask_generator = SamAutomaticMaskGenerator(sam)

    img_lists = df["image_name"]
    for idx, i in enumerate(tqdm(img_lists)):
        seg_path = os.path.join(img_path[:-6] + "image_segments/", i.split(".")[0])
        if os.path.exists(seg_path):
            continue

        image = cv2.imread(os.path.join(img_path, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(image)

        filter_and_save_segments(masks, image, img_path, df["description"][idx])

        # plt.imshow(image)
        # show_anns(masks, image, os.path.join(img_path, i))
        # masks = masks[:10]
        # show_masks_mine(masks, image, os.path.join(img_path, i))
