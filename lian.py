import os
from tqdm import tqdm

path = 'urbancross_data/images_target/Spain/image_segments/'
cnt = 0
for i in tqdm(os.listdir(path)):
    if len(os.listdir(os.path.join(path, i))) < 11:
        print(i)
        cnt += 1
        print(cnt)
    # import ipdb; ipdb.set_trace()


