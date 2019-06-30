import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class lstmposemachine(nn.Module):
    def __init__(self, out=7, stage=3):
        # 注意7类里面，每类输出3个值，x,y坐标，visible值
        super(lstmposemachine, self).__init__()
        # image: seq(5), 3, 270, 480
        # output label: 3, 13, seq(5)
        self.out=out
        self.stage=stage

        self.initnet = InitNet()

        self.convnet2 = ConvNet2()
        self.lstm = LSTM()
        self.convnet3 = ConvNet3()
        

        # self.testNet =  torch.nn.Sequential(
        #                     torch.nn.Linear(5*3*270*480, 100),
        #                     torch.nn.ReLU(),
        #                     torch.nn.Linear(100, 3*13*5)
        #                 )

    def InitNet():
        # convnet1(with dropout) and loss init
        # an additional ConvNet1
        
        return  torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,      # input height
                        out_channels=16,    # n_filters
                        kernel_size=5,      # filter size
                        stride=1,           # filter movement/step
                        padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                    ),      # output shape (16, 28, 28)
                    nn.ReLU(),    # activation
                    nn.MaxPool2d(kernel_size=2),
                    torch.nn.Dropout(0.5),
                )

    def ConvNet2():
        # handle the image,multi-layer CNN network
        return  torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,      # input height
                        out_channels=16,    # n_filters
                        kernel_size=5,      # filter size
                        stride=1,           # filter movement/step
                        padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                    ),      # output shape (16, 28, 28)
                    nn.ReLU(),    # activation
                    nn.MaxPool2d(kernel_size=2),
                )

    def ConvNet3():
        # handle lstm output
        return  torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=16,      # input height
                        out_channels=16,    # n_filters
                        kernel_size=5,      # filter size
                        stride=1,           # filter movement/step
                        padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                    ),      # output shape (16, 28, 28)
                    nn.ReLU(),    # activation
                    nn.MaxPool2d(kernel_size=2)
                )

                    
    def fuconLoss():
        return  torch.nn.Sequential(
                    torch.nn.Linear(16*120*140, 3*15*15), # 转成图片的大小
                )

    def LSTM():
        # LSTM
        return nn.LSTM(16, 16, 5)  #  ->   (input_size,hidden_size,num_layers)

    def forward(self, images):  
        # ininet
        #   repeat stages for frame

        # 每个阶段输入 为 经过图片经过convnet2的输出，central Gaussian map， 上一次loss的输出

        # 注意LSTM复用隐藏层

        return self.testNet(images.reshape(len(images),-1)).reshape(-1, 3, 13, 5)


def pck_score(predict, target, a, box):
    # box
    # return (7+1)
    # print(predict.shape)
    # print(target.shape)

    h , w = box
    bs, dindex, actionlen, seqlen = predict.shape

    beta = max(h,w)
    result = np.zeros(7+1)
    wrong = np.zeros(7+1)
    for batchsize in range(bs):
        # 每批
        for seq in range(seqlen):
            # 每张图片
            for action in range(actionlen):
                # 每个动作
                if (((predict[batchsize][2][action][seq] > 0.5) & (target[batchsize][2][action][seq] == 1) ) & \
                        (abs(predict[batchsize][1][action][seq] - target[batchsize][1][action][seq]) < beta) & \
                        (abs(predict[batchsize][0][action][seq] - target[batchsize][0][action][seq]) < beta) ) \
                    | ((predict[batchsize][2][action][seq] < 0.5) & (target[batchsize][2][action][seq] == 0) ):
                    result[action//2]+=1
                else:
                    wrong[action//2]+=1

    for action in range(len(result)-1):
        # print(result[action])
        # print(wrong[action])

        result[action] = result[action]/(result[action]+wrong[action])
        result[len(result)-1] += result[action]

    result[len(result)-1] /= len(result)-1
    return result