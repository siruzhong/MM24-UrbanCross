import os
import pandas as pd
import random

# country = "Finland"
# country = 'Spain'
country = 'Germany'
df = pd.read_csv(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/instructblip_generation_{country.lower()}_refine.csv")

img_name_list = df["image_name"].tolist()
random.shuffle(img_name_list)
# 划分比例
finetune_ratio = 0.2
finetune_val_ratio = 0.1
# val_ratio = 0.1
zeroshot_ratio = 0.7

# 计算划分后的大小
total_length = len(img_name_list)
finetune_length = int(finetune_ratio * total_length)
finetune_val_length = int(finetune_val_ratio * total_length)
zeroshot_length = int(zeroshot_ratio * total_length)
# val_length = int(val_ratio * total_length)
# test_length = int(test_ratio * total_length)

# 划分数据
finetune_list = img_name_list[:finetune_length]
finetune_val_list = img_name_list[
    finetune_length : finetune_length + finetune_val_length
]
zeroshot_list = img_name_list[finetune_length + finetune_val_length :]

stage = ["finetune", "finetune_val", "zeroshot"]
stage2list = {
    "finetune": finetune_list,
    "finetune_val": finetune_val_list,
    "zeroshot": zeroshot_list,
}

for st in stage:
    if not os.path.exists(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/{st}_list.txt"):
        # os.makedirs(f'urbancross_data/images_target/Finland/{st}_list.txt')
        file = open(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/{st}_list.txt", "w")
        file.close()

    with open(f"/hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target/{country}/{st}_list.txt", "w") as f:
        for i in stage2list[st]:
            f.write(i)
            f.write("\n")
