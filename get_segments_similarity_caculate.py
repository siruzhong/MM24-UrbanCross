import torch
import clip
from PIL import Image
import pandas as pd
from pathlib import Path

# 初始化pandas DataFrame来存储结果
new_rows = []  # 用于收集所有新行的列表
results_df = pd.DataFrame(columns=['Image Name', 'Segment Name', 'Similarity'])

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 函数：将描述文本分割并编码，然后取均值
def encode_and_average(description, model, preprocess, device):
    # 按照分号分割描述文本
    parts = description.split(';')
    # 移除空字符串
    parts = [part.strip() for part in parts if part.strip()]
    # 初始化存储向量的列表
    vectors = []
    # 对每部分单独处理
    for part in parts:
        text_inputs = clip.tokenize([part]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            vectors.append(text_features)
    # 计算向量的均值
    mean_vector = torch.mean(torch.stack(vectors), dim=0)
    return mean_vector

# 读取CSV文件中的描述
df = pd.read_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/dataset_rsicd.csv")

# 图像路径
img_path = "/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/image_segments"

# 遍历所有图像
for index, row in df[:2].iterrows():
    img_name = row['image_name'].split('.')[0]
    description = row['description']
    
    # 将描述文本编码并取均值
    text_features = encode_and_average(description, model, preprocess, device)
    
    # 获取分割图像，并排除不需要的文件
    segment_images = [img_file for img_file in Path(f"{img_path}/{img_name}").glob("*") 
                  if img_file.stem.endswith(('_',) + tuple(str(i) for i in range(50))) and
                  img_file.stem.split('_')[-1].isdigit() and
                  0 <= int(img_file.stem.split('_')[-1]) <= 49]
    # segment_images = [img_file for img_file in Path(f"{img_path}/{img_name}").glob("*") if "_0" <= img_file.stem[-2:] <= "_9"]
    for img_file in segment_images:
        # 图像转换为向量表示
        image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)

        # 计算图像与描述的相似度
        similarity = torch.cosine_similarity(text_features, image_features).cpu().numpy()
        
        # 打印相似度
        print(f"Similarity between {img_file.name} and text: {similarity[0]}")
        
        # 将结果添加到DataFrame
        new_row = {'Image Name': img_name, 'Segment Name': img_file.name, 'Similarity': similarity[0]}
        new_rows.append(new_row)
        
        # # 基于相似度过滤
        # if similarity > 0.3:
        #     print(f"Keeping {img_file} with similarity {similarity}")
        # else:
        #     print(f"Removing {img_file} due to low similarity {similarity}")
        #     img_file.unlink()  # 删除图像文件
        
# 打印每个图像的分割和相应的相似度表格
results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)

# 可选：保存结果到CSV文件
results_df.to_csv("/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/similarity_results.csv", index=False)
