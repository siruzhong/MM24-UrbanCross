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

def show_anns(anns, ori_img, img_path):
    img_name = img_path.split('/')[-1]
    seg_path = img_path.replace('/images/', '/image_segments/').split('.')[0]
    # import ipdb; ipdb.set_trace()
        
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    #anns就是整个masks的list传过来了
    if len(anns) == 0:
        return
    #按面积从大到小排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    #img [256,256,4]
    # import ipdb;ipdb.set_trace()
    img[:,:,3] = 0
    for idx,ann in enumerate(sorted_anns):
        m = ann['segmentation']
        #m是[534,800], 是一个true和false的numpy array
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        # plt.clf()
        # plt.imshow(ori_img)
        # plt.imshow(img)
        # plt.savefig('pra'+str(idx)+'.jpg')
    ax.imshow(img)
    # import ipdb;ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    plt.axis('off')
    plt.savefig(
            os.path.join(seg_path, img_name),
            bbox_inches='tight',
            pad_inches=0
    )
    
def show_masks_mine(anns, ori_img, img_path):
    # import ipdb;ipdb.set_trace()
    # seg_path = img_path.replace('images', 'image_segments').split('.')[0]
    img_name = img_path.split('/')[-1].split('.')[0]
    # img_name = img_path.split('/')[-1]
    seg_path = img_path.replace('/images/', '/image_segments/').split('.')[0]
    
    #anns就是整个masks的list传过来了
    if len(anns) == 0:
        return
    #按面积从大到小排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    masked_img = np.copy(ori_img)
    # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    #img [256,256,4]
    
    # img[:,:,3] = 0
    for idx,ann in enumerate(sorted_anns):
        m = ann['segmentation']
        masked_img = np.copy(ori_img)
        masked_img[m!=1] = [255,255,255]
        #m是[534,800], 是一个true和false的numpy array
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # ori_img[]
        # img[m] = color_mask
        # import ipdb;ipdb.set_trace()
        plt.clf()
        plt.imshow(masked_img)
        # plt.imshow(img)
        plt.axis('off')
        
        # img_path = img_path.replace('images_rgb', 'image_segments')
        plt.savefig(
                    os.path.join(seg_path, img_name+'_'+str(idx)+'.jpg'),
                    bbox_inches='tight',
                    pad_inches=0
        )
    # ax.imshow(img)


if __name__ == '__main__':
    # img_path = 'urbancross_data/images_target/Finland/images'
    country = 'Germany'
    img_path = f'urbancross_data/images_target/{country}/images'
    df = pd.read_csv(f'urbancross_data/instructblip_generation_with_tag/instructblip_generation_{country.lower()}_refine.csv')
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cuda')

    mask_generator = SamAutomaticMaskGenerator(sam)

    # import ipdb; ipdb.set_trace()
    # image = cv2.imread('notebooks/images/dog.jpg')
    img_lists = df['image_name']
    # import ipdb; ipdb.set_trace()
    # for idx, row in tqdm(df.iterrows()):
    # for idx, i in enumerate(tqdm(img_lists[::-1])):
    # for idx, i in enumerate(tqdm(img_lists[::-1][7000:])):
    # for idx, i in enumerate(tqdm(img_lists[::-1][14000:])):
    # for idx, i in enumerate(tqdm(img_lists[::-1][21000:])):
    # for idx, i in enumerate(tqdm(img_lists[::-1][28000:])):
    # for idx, i in enumerate(tqdm(img_lists[::-1][35000:])):
    for idx, i in enumerate(tqdm(img_lists)):
    # for idx, i in enumerate(tqdm(img_lists[10000:])):
    # for idx, i in enumerate(tqdm(img_lists[20000:])):
    # for idx, i in enumerate(tqdm(img_lists[30000:])):
    # for idx, i in enumerate(tqdm(img_lists[40000:])):
    # for idx, i in enumerate(tqdm(img_lists[50000:])):
    # for idx, i in enumerate(tqdm(img_lists[60000:])):
    # for idx, i in enumerate(tqdm(img_lists[80000:])):
    # for idx, i in enumerate(tqdm(img_lists[100000:])):
    # for idx, i in enumerate(tqdm(img_lists[120000:])):
    # for idx, i in enumerate(tqdm(img_lists[164500:])):
        seg_path = os.path.join(img_path[:-6] + 'image_segments/', i.split('.')[0])
        # seg_path = os.path.join(img_path[:-6] + 'image_segments/', img_name)
        if os.path.exists(seg_path):
            continue
        # import ipdb;ipdb.set_trace()
        # seg_path = os.path.join(img_path.replace('images_rgb', 'image_segments').split('.')[0], i.split('.')[0])
        # image = cv2.imread('../SWAN-pytorch-main/rs_data/rsitmd/images_rgb/airport_2.jpg')
        # print(os.path.join(img_path, i))
        image = cv2.imread(os.path.join(img_path, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image [256,256,3]
        # import ipdb;ipdb.set_trace()
        # plt.figure(figsize=(20,20))
        # plt.imshow(image)
        # plt.axis('off')
        # plt.savefig('pra.jpg')


        masks = mask_generator.generate(image)

        #masks是list
        #len(masks) 是66
        #每个mask是个dict
        # ipdb> masks[0].keys()
        # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])

        plt.imshow(image)
        show_anns(masks, image, os.path.join(img_path, i))
        masks = masks[:10]
        show_masks_mine(masks, image, os.path.join(img_path, i))