import os
import pandas as pd
import random

# Specify the country of interest for dataset processing
# Uncomment the desired country or modify the value as needed
# country = "Finland"
# country = 'Spain'
country = 'Germany'

# Load the CSV file containing image metadata for the specified country
# The CSV file is expected to be in the UrbanCross dataset directory
# The filename should match the instructblip_generation_<country>_refine.csv pattern
df = pd.read_csv(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/{country}/instructblip_generation_{country.lower()}_refine.csv")

# Convert the "image_name" column to a list
# This list will contain the names of all images in the dataset
img_name_list = df["image_name"].tolist()

# Shuffle the image list to ensure random assignment to different splits
random.shuffle(img_name_list)

# Define split ratios for different stages of dataset usage
# finetune_ratio: Proportion of images for fine-tuning the model
# finetune_val_ratio: Proportion of images for fine-tuning validation
# zeroshot_ratio: Proportion of images for zero-shot evaluation
finetune_ratio = 0.2
finetune_val_ratio = 0.1
zeroshot_ratio = 0.7

# Calculate the number of images for each split based on the specified ratios
total_length = len(img_name_list)
finetune_length = int(finetune_ratio * total_length)
finetune_val_length = int(finetune_val_ratio * total_length)
zeroshot_length = int(zeroshot_ratio * total_length)

# Split the shuffled list of images based on the calculated lengths
finetune_list = img_name_list[:finetune_length]  # First portion for fine-tuning
finetune_val_list = img_name_list[
    finetune_length : finetune_length + finetune_val_length
]  # Next portion for fine-tuning validation
zeroshot_list = img_name_list[finetune_length + finetune_val_length :]  # Remaining for zero-shot evaluation

# Define the stages and create a mapping from stage name to image list
stage = ["finetune", "finetune_val", "zeroshot"]
stage2list = {
    "finetune": finetune_list,
    "finetune_val": finetune_val_list,
    "zeroshot": zeroshot_list,
}

# Create the split files for each stage if they do not exist
# Write the corresponding image names into separate text files
for st in stage:
    # Check if the file already exists, if not, create an empty file
    if not os.path.exists(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/{country}/{st}_list.txt"):
        file = open(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/{country}/{st}_list.txt", "w")
        file.close()

    # Open the file in write mode and write each image name in the list to the file
    with open(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/{country}/{st}_list.txt", "w") as f:
        for i in stage2list[st]:
            f.write(i)
            f.write("\n")