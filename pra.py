import os
import pandas as pd
import shutil
from tqdm import tqdm


df = pd.read_csv('urbancross_data/images_target/Finland/captions_old.csv')
img_list = os.listdir('urbancross_data/images_target/Finland/images')

df = df[df['image_name'].isin(img_list)]
df.to_csv('urbancross_data/images_target/Finland/captions.csv', index=False)
import ipdb;ipdb.set_trace()
df_img_list = df['image_name'].tolist()
cnt = 0
for i in tqdm(df_img_list):
    if i not in img_list:
        # print(i)
        cnt += 1
        print(cnt)
    


# for i in tqdm(df['image_name']):
    
#     shutil.copy(f'urbancross_data/images_target/Spain/images/{i}', f'urbancross_data/images_target/Spain/images_top30/{i}')
#     # print(i)
    
# import ipdb; ipdb.set_trace()
