import os
import pandas as pd
import random

# country = 'Finland'
country = "Spain"
df = pd.read_csv(f"{country}/captions.csv")

img_name_list = df["image_name"].tolist()
random.shuffle(img_name_list)

# 划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# 计算划分后的大小
total_length = len(img_name_list)
train_length = int(train_ratio * total_length)
val_length = int(val_ratio * total_length)
test_length = int(test_ratio * total_length)

# 划分数据
train_list = img_name_list[:train_length]
val_list = img_name_list[train_length : train_length + val_length]
test_list = img_name_list[train_length + val_length :]

stage = ["train", "val", "test"]
stage2list = {"train": train_list, "val": val_list, "test": test_list}

for st in stage:
    if not os.path.exists(f"{country}/{st}_list.txt"):
        # os.makedirs(f'urbancross_data/images_target/Finland/{st}_list.txt')
        file = open(f"{country}/{st}_list.txt", "w")
        file.close()

    with open(f"{country}/{st}_list.txt", "w") as f:
        for i in stage2list[st]:
            f.write(i)
            f.write("\n")
