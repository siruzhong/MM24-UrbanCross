import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm 

transform_ = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

img_path = 'rs_data/rsitmd/images'
# new_img_path
for i in tqdm(os.listdir(img_path)):
    image = Image.open(os.path.join(img_path , i)).convert('RGB')
    # import ipdb;ipdb.set_trace()
    # image = transform_(image)  # torch.Size([3, 256, 256])

    image.save(os.path.join(img_path+'_rgb', i.replace('.tif','.jpg')))
