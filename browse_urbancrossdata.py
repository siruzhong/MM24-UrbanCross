import pandas as pd

# df = pd.read_csv('urbancross_data/instructblip_generation_with_tag/instructblip_generation_spain_refine.csv')

df = pd.read_csv('urbancross_data/images_target/Spain/captions_top30.csv')

for index, row in df.iterrows():
    # 打印每一行的内容
    print(row['title_multi_objects'])
    import ipdb;ipdb.set_trace()
    print()
    #print(row['column1'], row['column2'], row['column3'])
    
