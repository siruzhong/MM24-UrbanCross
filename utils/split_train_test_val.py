import os
import pandas as pd
import random

# Define the country name to use for data loading. Uncomment the desired country.
# country = 'Finland'
country = "Spain"

# Load the CSV file containing image captions.
df = pd.read_csv(f"{country}/captions.csv")

# Get the list of image names and shuffle the list to randomize the data.
img_name_list = df["image_name"].tolist()
random.shuffle(img_name_list)

# Define the split ratios for train, validation, and test datasets.
train_ratio = 0.7  # 70% of the data for training
val_ratio = 0.1    # 10% of the data for validation
test_ratio = 0.2   # 20% of the data for testing

# Calculate the sizes of each dataset split.
total_length = len(img_name_list)
train_length = int(train_ratio * total_length)
val_length = int(val_ratio * total_length)
test_length = int(test_ratio * total_length)

# Split the data into train, validation, and test sets.
train_list = img_name_list[:train_length]
val_list = img_name_list[train_length : train_length + val_length]
test_list = img_name_list[train_length + val_length :]

# Define the different dataset stages.
stage = ["train", "val", "test"]
# Create a dictionary to map the dataset stages to the corresponding image lists.
stage2list = {"train": train_list, "val": val_list, "test": test_list}

# Iterate over each stage to create the corresponding list files.
for st in stage:
    # Check if the file does not exist and create it if necessary.
    if not os.path.exists(f"{country}/{st}_list.txt"):
        # Create an empty file for the stage.
        file = open(f"{country}/{st}_list.txt", "w")
        file.close()

    # Open the file in write mode and write each image name into the file.
    with open(f"{country}/{st}_list.txt", "w") as f:
        for i in stage2list[st]:
            f.write(i)
            f.write("\n")