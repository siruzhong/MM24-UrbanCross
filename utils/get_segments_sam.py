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

# Adding the parent directory to system path
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Load CLIP model to compute similarities
# Depending on GPU availability, the model is loaded onto either GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def compute_similarity(image_path, text, device=device, clip_model=clip_model, preprocess=preprocess):
    """
    Calculate similarity between an image and a text description. Long texts are split into smaller parts for processing.
    
    Args:
        image_path (str): Path to the image.
        text (str): Text description to be compared with the image.
        device (str): The device to run computations on (either 'cpu' or 'cuda').
        clip_model: The loaded CLIP model to encode images and texts.
        preprocess: Preprocessing function for input images.

    Returns:
        float: Cosine similarity between the image and text features.
    """
    # Split the text using regex to handle different punctuation marks like semicolons, commas, and periods
    text_parts = re.split(r'[;,.]', text)
    text_parts = [part.strip() for part in text_parts if part.strip()]
    text_features_list = []
    
    # Encode each part of the text separately
    for part in text_parts:
        try:
            text_inputs = clip.tokenize([part]).to(device)
            with torch.no_grad():
                text_features_part = clip_model.encode_text(text_inputs)
                text_features_list.append(text_features_part)
        except RuntimeError as e:
            print(f"Error processing text: {part}. Error: {e}")
            return None
    
    # Calculate the average of the encoded text features
    if not text_features_list:
        return None
    text_features_avg = torch.mean(torch.stack(text_features_list), dim=0)
    
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    # Calculate cosine similarity between the image and averaged text features
    similarity = torch.cosine_similarity(text_features_avg, image_features, dim=1).cpu().numpy()[0]
    return similarity


def all_segments_exist(seg_path, num_segments_expected=10):
    """
    Check if all the expected segmented images already exist.
    
    Args:
        seg_path (str): Directory path for the segmented images.
        num_segments_expected (int): Expected number of segments.
    
    Returns:
        bool: True if all segmented images exist, False otherwise.
    """
    # Verify if the number of existing segments is equal to or greater than the expected number
    existing_segments = [f for f in os.listdir(seg_path) if os.path.isfile(os.path.join(seg_path, f))]
    return len(existing_segments) >= num_segments_expected


def show_masks_mine(anns, ori_img, img_path, description):
    """
    Modified function to compute the similarity between each segmented image and a text description, and save only the top 10 segments based on similarity.
    
    Args:
        anns (list): List of segmentation masks.
        ori_img (ndarray): The original image array.
        img_path (str): Path to the original image.
        description (str): Text description to compute similarity against.
    """
    # Extract the image name from the image path
    img_name = img_path.split("/")[-1].split(".")[0]
    # Define the path for storing segments
    seg_path = img_path.replace("/images/", "/image_segments_new/").split(".")[0]
    
    # If the segment directory exists, check if all segments already exist and skip
    if os.path.exists(seg_path):
        if all_segments_exist(seg_path):
            print(f"All segments for {img_name} already exist. Skipping.")
            return
    
    # Ensure the target directory exists
    if not os.path.exists(seg_path):
        os.makedirs(seg_path, exist_ok=True)  # Create the directory and any necessary parent directories

    if len(anns) == 0:
        return

    # Sort annotations based on area size in descending order
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    
    # List to store similarities and corresponding paths
    similarities = []
    
    # Process a maximum of the top 50 segments
    for idx, ann in enumerate(sorted_anns[:50]):
        m = ann["segmentation"]
        masked_img = np.copy(ori_img)
        masked_img[m != 1] = [255, 255, 255]
        
        # Save each segment as an image
        segment_img_path = os.path.join(seg_path, f"{img_name}_{idx}.jpg")
        plt.imsave(segment_img_path, masked_img)

        # Compute similarity with the provided description
        similarity = compute_similarity(segment_img_path, description)
        if similarity is not None:
            similarities.append((similarity, segment_img_path))
    
    # Keep only the top 10 segments based on similarity
    top_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]
    for similarity, path in top_similarities:
        print(f"Kept {path} with similarity {similarity}")

    # Remove other segments
    for similarity, path in similarities:
        if path not in [x[1] for x in top_similarities]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    # Set paths for images and CSV file
    img_path = "/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/Germany/images"
    df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/Germany/instructblip_generation_germany_refine.csv")

    # Load the segmentation model checkpoint
    sam_checkpoint = "/hpc2hdd/home/szhong691/zsr/projects/segment-anything/segment_anything/checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    # List of image names from the CSV file
    img_lists = df["image_name"]
    failed_rows = []

    # Iterate over the dataset to process each image and description pair
    for idx, row in tqdm(list(df.iterrows())[101468:], total=df.shape[0]):
        image_name = row['image_name']
        description = row['description']  # Ensure that there is a description column in the CSV
        image_path = os.path.join(img_path, image_name)
        
        # If the image does not exist, skip to the next iteration
        if not os.path.exists(image_path):
            continue
        
        # Load the image and convert it to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        
        # Try to generate masks and compute similarities
        try:
            show_masks_mine(masks, image, image_path, description)
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            failed_rows.append(idx)
    
    # Remove the rows that failed processing from the dataframe
    for idx in sorted(failed_rows, reverse=True):
        df.drop(idx, inplace=True)
    
    # Save the updated CSV file
    df.to_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/Germany/instructblip_generation_germany_refine_fixed.csv", index=False)