import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import sys
import re
from tqdm import tqdm

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def compute_similarity(image_path, text, device=device, clip_model=clip_model, preprocess=preprocess):
    """
    计算图像与文本描述的相似度，对长文本进行分割处理。
    """
    # 使用正则表达式分割文本，支持分号、逗号、句号
    text_parts = re.split(r'[;,.]', text)
    text_parts = [part.strip() for part in text_parts if part.strip()]
    text_features_list = []
    
    # 对每个部分单独进行编码
    for part in text_parts:
        text_inputs = clip.tokenize([part]).to(device)
        with torch.no_grad():
            text_features_part = clip_model.encode_text(text_inputs)
            text_features_list.append(text_features_part)
    
    # 计算所有部分编码的平均值
    text_features_avg = torch.mean(torch.stack(text_features_list), dim=0)
    
    # 加载并处理图像
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    # 计算图像与平均文本特征的相似度
    similarity = torch.cosine_similarity(text_features_avg, image_features, dim=1).cpu().numpy()[0]
    return similarity


def all_segments_exist(seg_path, num_segments_expected=10):
    """
    检查是否所有的图像分割都已经存在。
    
    Args:
        seg_path (str): 分割图像的目录路径。
        num_segments_expected (int): 预期的分割图像数量。
    
    Returns:
        bool: 如果所有分割图像都存在，则返回True，否则返回False。
    """
    # 检查存在的分割图像数量是否等于预期的数量
    existing_segments = [f for f in os.listdir(seg_path) if os.path.isfile(os.path.join(seg_path, f))]
    return len(existing_segments) >= num_segments_expected


def show_masks_mine(anns, ori_img, img_path, description):
    """
    修改后的函数，计算分割图像与文本描述的相似度，并只保存相似度最高的前10个分割。
    """
    img_name = img_path.split("/")[-1].split(".")[0]
    seg_path = img_path.replace("/images/", "/image_segments_new/").split(".")[0]
    
    if os.path.exists(seg_path):
        # 如果分割图像已经存在，跳过当前图像
        if all_segments_exist(seg_path):
            print(f"All segments for {img_name} already exist. Skipping.")
            return
    
    # 确保目标目录存在
    if not os.path.exists(seg_path):
        os.makedirs(seg_path, exist_ok=True)  # 创建目录和所有必需的父目录

    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    
    # 存储相似度和路径
    similarities = []
    
    for idx, ann in enumerate(sorted_anns[:50]):  # 假设最多处理前50个分割
        m = ann["segmentation"]
        masked_img = np.copy(ori_img)
        masked_img[m != 1] = [255, 255, 255]
        
        segment_img_path = os.path.join(seg_path, f"{img_name}_{idx}.jpg")
        plt.imsave(segment_img_path, masked_img)

        similarity = compute_similarity(segment_img_path, description)
        similarities.append((similarity, segment_img_path))
    
    # 保留相似度最高的前10个分割
    for similarity, path in sorted(similarities, key=lambda x: x[0], reverse=True)[:10]:
        print(f"Kept {path} with similarity {similarity}")

    # 删除其他分割
    for similarity, path in sorted(similarities, key=lambda x: x[0], reverse=True)[10:]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    # img_path = "/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/Finland/images"
    # df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/Finland/instructblip_generation_finland_refine.csv")
    img_path = "/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/Germany/images"
    df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/Germany/instructblip_generation_germany_refine.csv")

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    img_lists = df["image_name"]
    # for idx, row in tqdm(reversed(list(df.iterrows())[10000:]), total=df.shape[0]):
    for idx, row in tqdm(list(df.iterrows())[5475:], total=df.shape[0]):
        image_name = row['image_name']
        description = row['description']  # 确保CSV中有描述的列
        image_path = os.path.join(img_path, image_name)
        
        if not os.path.exists(image_path):
            continue
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        
        # 直接在这里调用修改后的函数
        show_masks_mine(masks, image, image_path, description)
