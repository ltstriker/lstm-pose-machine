'''
only used for penn_action datasets
'''
from __future__ import print_function
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Penn_Data(Dataset):
    def __init__(self, data_dir='/BIGDATA1/nsccgz_yfdu_5/liangt/LSTMPoseMachine/lstm_pm_pytorch/Penn_Action/', train=True, transform=None):

        self.input_h = 368
        self.input_w = 368
        self.map_h = 45
        self.map_w = 45

        self.parts_num = 13
        self.seqTrain = 5

        self.gaussian_sigma = 21

        self.transform = transform

        self.train = train
        if self.train is True:
            self.data_dir = data_dir + 'train/'
        else:
            self.data_dir = data_dir + 'test/'

        self.frames_data = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.frames_data)  # number of videos in train or test

    def __getitem__(self, idx):  # get a video sequence
        '''

        :param idx:
        :return:
            images:     Tensor    seqtrain * 3 * width * height
            label_map:  Tensor    46 * 46 * (class+1) * seqtrain
            center_map: Tensor    1 * 368 * 368
        '''
        frames = self.frames_data[idx]
        data = np.load(os.path.join(self.data_dir, frames)).item()

        images, label_map = self.transformation_penn(data, boxsize=self.input_w, parts_num=13,
                                                               train=self.train)

        return images, label_map

    def transformation_penn(self, data, boxsize=368, parts_num=13, train=True):
        '''
        :param data:
        :param boxsize:
        :param parts_num:
        :param seqTrain:
        :param train:
        :return:
        images tensor seq
        '''
        nframes = data['nframes']                           # 151
        framespath = data['framepath']
        dim = data['dimensions']                            # [360, 480]
        x = data['x']                                       # 151 * 13
        y = data['y']                                       # 151 * 13
        visibility = data['visibility']                     # 151 * 13

        start_index = np.random.randint(0, nframes - 1 - self.seqTrain + 1)  #

        images = torch.zeros(self.seqTrain, 3, 270, 480)  # tensor seqTrain * 3 * 368 * 368
        label = np.zeros((3, parts_num, self.seqTrain))     # numpy 3

        for i in range(self.seqTrain):
            # read image
            img_path = os.path.join(framespath,'%06d' % (start_index + i + 1) + '.jpg')
            img = Image.open(img_path)  # Image
            # print(img.size)
            images[i, :, :, :] = self.transform(img)  # store image

            # read label
            label[0, :, i] = x[start_index + i]
            label[1, :, i] = y[start_index + i]
            label[2, :, i] = visibility[start_index + i]  # 1 * 13

        return images, label

# transform1 = transforms.Compose([
#     transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
#     ]
# )


transform = transforms.Compose([transforms.RandomCrop([ 270, 480],50),transforms.ToTensor()])

# # # test case
data = Penn_Data(data_dir='Penn_Action/', transform=transform)
images, label_map = data[1]

# print(data[1])
print(images.shape)

print(label_map.shape)
