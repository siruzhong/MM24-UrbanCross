import os
import pandas as pd
import shutil
from tqdm import tqdm

df = pd.read_csv('/home/yuxuan/zsr/projects/dataset/SkyScript/images_target/Spain/captions_top30.csv')

for i in tqdm(df['image_name']):
    
    shutil.copy(f'urbancross_data/images_target/Spain/images/{i}', f'urbancross_data/images_target/Spain/images_top30/{i}')
    # print(i)
    
# import ipdb; ipdb.set_trace()
