import torch.nn as nn
import torch.nn.functional as F
import torch



class lstmposemachine(nn.Module):
    def __init__(self, out=7, stage=3):
        # 注意7类里面，每类输出3个值，x,y坐标，visible值
        super(lstmposemachine, self).__init__()
        self.out=out
        self.stage=stage


    def InitNet():
        # convnet1(with dropout) and loss init
        # an additional ConvNet1
        pass

    def ConvNet2():
        # handle the image,multi-layer CNN network
        pass

    def ConvNet3():
        # handle lstm output
        pass

    def LSTM():
        # LSTM
        pass

    def forward(self, images, center_map):  
        # ininet
        #   repeat stages for frame
        pass


def pck_score(predict, target, a, box):
    pass