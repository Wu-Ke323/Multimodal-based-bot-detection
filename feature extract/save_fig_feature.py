import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torchvision.models import ResNet50_Weights
import imghdr
import shutil

# 加载预训练的ResNet模型
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 768)
# 设置模型为评估模式
resnet.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 用户图片目录
user_directory = '/root/data/10000_11826photo/'
# 存储每个用户的特征向量列表
user_features = []

# 遍历用户图片目录
path_list = os.listdir(user_directory)
path_list.sort(key=lambda x:int(x[0:-6]))
#print(path_list)
for user_folder in path_list:
   # 拼接用户图片目录的完整路径
    user_folder_path = os.path.join(user_directory, user_folder)
    print(user_folder_path)
    #检查路径是否为目录
    if os.path.isdir(user_folder_path):
        # 存储当前用户的特征向量列表
        user_vector = []
        
        if len(os.listdir(user_folder_path)) == 0:
            user_vector = np.zeros(768)
            print('该用户无图片数据')
            user_features.append(user_vector)
            continue

        # 遍历当前用户图片目录下的所有图片
        for filename in os.listdir(user_folder_path):
            # 拼接图片文件的完整路径
            image_path = os.path.join(user_folder_path, filename)
            if os.path.splitext(image_path)[-1] in (".jpg", ".png", ".jpeg", ".gif"):
                #print(image_path)
                if imghdr.what(image_path) == None:
                    continue
                # 加载图像
                image = Image.open(image_path)
                # 预处理图像
                # print(image.split())
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = transform(image)
                # 添加一个维度作为批处理维度
                image = image.unsqueeze(0)

                # 前向传播获得特征表示
                image_features = resnet(image)
                # 提取特征
                image_features = image_features.squeeze().detach().numpy()
                # 将特征添加到当前用户的特征向量列表中
                user_vector.append(image_features)

        # 对当前用户的特征向量列表进行平均得到一个代表该用户的特征向量
        if user_vector:
            user_vector = np.mean(user_vector, axis=0)
            print(user_vector.size)
        else:
            user_vector = np.zeros(768)
            print("用户图片数据均损坏")
        # 将特征向量添加到存储所有用户特征向量的列表中
        user_features.append(user_vector)

# 将存储所有用户特征向量的列表转换为NumPy数组
user_features = np.array(user_features)
user_features = torch.tensor(user_features).float()

torch.save(user_features, '/root/data/photo_feature/10000_11826photo.pt')
temp = torch.load('/root/data/photo_feature/10000_11826photo.pt')
print(temp.size())
