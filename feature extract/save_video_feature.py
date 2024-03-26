import os
import sys
import subprocess
import json
import torchvision
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from torch.autograd import Variable
from timesformer.models.transforms import *
from timesformer.models.vit import TimeSformer

device = torch.device("cuda:0")


def get_input(image_path):
    prefix = '{:05d}.jpg'
    feat_path = image_path
    video_data = {}
    images = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_params = {
        "side_size": 256,
        "crop_size": 224,
        "num_segments": 8,
        "sampling_rate": 5
    }
    transform_val = torchvision.transforms.Compose([
        GroupScale(int(transform_params["side_size"])),
        GroupCenterCrop(transform_params["crop_size"]),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(mean, std),
    ])
    frame_list = os.listdir(feat_path)
    average_duration = len(frame_list) // transform_params["num_segments"]
    # offests为采样坐标
    offsets = np.array(
        [int(average_duration / 2.0 + average_duration * x) for x in range(transform_params["num_segments"])])
    offsets = offsets + 1
    for seg_ind in offsets:
        p = int(seg_ind)
        seg_imgs = Image.open(os.path.join(feat_path, prefix.format(p))).convert('RGB')
        images.append(seg_imgs)
    video_data = transform_val(images)
    video_data = video_data.view((-1, transform_params["num_segments"]) + video_data.size()[1:])
    out = Variable(video_data)

    return out


def extract(modal, data, out_image_dir):
    output = {}
    if modal == 'video':
        # =================模型建立======================
        model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                            pretrained_model='/root/TimeSformer/TimeSformer_divST_8x32_224_K400.pyth')

        model = model.eval().to(device)
        # print(model)

        # =================视频抽帧======================
        video_name = data.split('/')[-1].split('.')[0]
        out_image_path = os.path.join(out_image_dir, video_name)
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
        cmd = 'ffmpeg -i \"{}\" -r 1 -q:v 2 -f image2 \"{}/%05d.jpg\"'.format(data, out_image_path)
        #print(cmd)
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # =================提取特征======================
        if len(os.listdir(out_image_path)) == 0:
            print(out_image_path, '为空')
            out = torch.zeros(768).float()
        else:
            model_input = get_input(out_image_path).unsqueeze(0).to(device)
            #print(model_input.shape)
            out = model(model_input, )
            out = out.squeeze(0)
            out = out.cpu().detach().numpy()

    return out


#1_1999
'''outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/1_1999'
for i in range(1, 2000):
    video_path = '/root/data/1_1999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print('读取：', video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/1_1999.pt')

#2000_3999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/2000_3999'
for i in range(2000,4000):
    video_path = '/root/data/2000_3999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/2000_3999.pt')'''

#4000_4999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/4000_4999'
for i in range(4000,5000):
    video_path = '/root/data/4000_4999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/4000_4999.pt')

#5000_5999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/5000_5999'
for i in range(5000,6000):
    video_path = '/root/data/5000_5999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/5000_5999.pt')

#6000_6999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/6000_6999'
for i in range(6000,7000):
    video_path = '/root/data/6000_6999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/6000_6999.pt')

#7000_7999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/7000_7999'
for i in range(7000,8000):
    video_path = '/root/data/7000_7999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/7000_7999.pt')

#8000_8999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/8000_8999'
for i in range(8000,9000):
    video_path = '/root/data/8000_8999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/8000_8999.pt')

#9000_9999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/9000_9999'
for i in range(9000,10000):
    video_path = '/root/data/9000_9999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/9000_9999.pt')

#10000_10499
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/10000_10499'
for i in range(10000,10500):
    video_path = '/root/data/10000_10499/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/10000_10499.pt')

#10500_10999
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/10500_10999'
for i in range(10500,11000):
    video_path = '/root/data/10500_10999/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/10500_10999.pt')

#11000_11499
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/11000_11499'
for i in range(11000,11500):
    video_path = '/root/data/11000_11499/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/11000_11499.pt')

#11500_11826
outs = []
out_image_dir = '/root/video_feature/extract_frame/video10_feature/11500_11826'
for i in range(11500,11827):
    video_path = '/root/data/11500_11826/'+str(i)+'_video/'+str(i)+'_10.mp4'
    print(video_path)
    if os.path.exists(video_path):
        modal = 'video'
        out = extract(modal, video_path, out_image_dir)
        outs.append(out)
    else:
        print('%d 用户不存在视频10数据' %i)
        out = np.zeros(768)
        outs.append(out)

outs = np.array(outs)
outs = torch.tensor(outs)
print(outs.size())
torch.save(outs, '/root/video_feature/video10_feature/11500_11826.pt')