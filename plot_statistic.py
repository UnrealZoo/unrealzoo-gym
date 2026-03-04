import json
import os
from collections import defaultdict
from typing import DefaultDict

import matplotlib.pyplot as plt

# 设置文件夹路径
folder_path = 'statistic'

# 初始化字典来存储面积数据
area_data: DefaultDict[str, int] = defaultdict(int)
size_data: DefaultDict[str, int] = defaultdict(int)
height_data: DefaultDict[str, int] = defaultdict(int)
object_data: DefaultDict[str, int] = defaultdict(int)

# 遍历文件夹中的所有文件
categories_area = ['0-999', '1000-9999', '10000-99999', '100000-999999']
categories_size = ['0-9999', '10000-99999', '100000-999999', '1000000-9999999', '10000000+']
categories_height = ['0-3', '3-10', '11-30', '31-100', '100+']
categories_objnum = ['0-100', '100-500', '500-1000', '1000-3000', '3000+']
for key in categories_area:
    area_data[key] = 0
for key in categories_size:
    size_data[key] = 0
for key in categories_height:
    height_data[key] = 0
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.json
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # 打开并读取JSON文件
        with open(file_path, 'r') as file:
            data = json.load(file)

            # 假设每个JSON文件中都有一个'Area'键
            # 将面积值添加到字典中
            # 根据面积值确定其所属的区间
            if 'area' in data:
                area = data['area']
                if area < 1000: # 31*31 = 961
                    category = '0-999'
                elif area < 10000: # 100*100 = 10000
                    category = '1000-9999'
                elif area < 1000000: # 316*316 = 99856
                    category = '10000-99999'
                elif area < 10000000: # 1000*1000 = 1000000
                    category = '100000-999999'
                else:
                    category = '1000000+'
                    print(data['env_name'], data['area'])
                area_data[category] += 1
            if 'size' in data:
                size = data['size']
                if size < 10000:
                    category = '0-9999'
                elif size < 100000:
                    category = '10000-99999'
                elif size < 1000000:
                    category = '100000-999999'
                elif size < 10000000:
                    category = '1000000-9999999'
                else:
                    category = '10000000+'
                size_data[category] += 1
            if 'bbox' in data:
                height = data['bbox'][2]
                if height <= 3:
                    category = '0-3'
                elif height <= 10:
                    category = '3-10'
                elif height <= 30:
                    category = '11-30'
                elif height <= 100:
                    category = '31-100'
                else:
                    category = '100+'
                height_data[category] += 1

            if 'object_num' in data:
                obj_num = data['object_num']
                if obj_num <= 100:
                    category = '0-100'
                elif obj_num <= 500:
                    category = '100-500'
                elif obj_num <= 1000:
                    category = '500-1000'
                elif obj_num <= 3000:
                    category = '1000-3000'
                else:
                    category = '3000+'
                object_data[category] += 1

# 准备数据用于绘图
# categories = list(area_data.keys())
# counts = list(area_data.values())
print(area_data, size_data, height_data)

def plot_bar_chart(categories, data, title, xlabel, ylabel):
    colormap = plt.get_cmap('viridis')
    counts = [data[category] for category in categories if category in data]
    plt.bar(categories, counts, color=[colormap(i / len(counts)) for i in range(len(counts))])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


plot_bar_chart(categories_area, area_data, 'Distribution of Area Categories', 'Area Categories', 'Count')
plot_bar_chart(categories_size, size_data, 'Distribution of Size Categories', 'Size Categories', 'Count')
plot_bar_chart(categories_height, height_data, 'Distribution of Height Categories', 'Height Categories', 'Count')
plot_bar_chart(categories_objnum, object_data, 'Distribution of Object Number Categories', 'Object Number Categories', 'Count')
# # 创建柱状图，使用颜色映射
# counts = [area_data[category] for category in categories if category in area_data]
# bars = plt.bar(categories, counts, color=[colormap(i / len(counts)) for i in range(len(counts))])
#
# # 添加标题和标签
# plt.xlabel('Area Categories')
# plt.ylabel('Count')
# plt.title('Distribution of Area Categories')
#
#
# # 显示图表
# plt.show()