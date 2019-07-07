import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import numpy as np
from scipy.stats import multivariate_normal


class lstmposemachine(nn.Module):
    def __init__(self, out=7, stage=3, device=torch.device("cpu")):
        # 注意7类里面，每类输出3个值，x,y坐标，visible值
        super(lstmposemachine, self).__init__()
        # image: seq(5), 3, 270, 480
        # output label: 3, 13, seq(5)
        self.out=out
        self.stage=stage

        self.initnet = self.InitNet()

        self.convnet2 = self.ConvNet2()
        self.lstm = self.LSTM()
        self.convnet3 = self.ConvNet3()
        
        self.device = device

        self.lossnet = self.fuconLoss()

        
        self.resultnet = torch.nn.Linear(120 * 67, 3*13*stage) # 转成图片的大小
        # self.testNet =  torch.nn.Sequential(
        #                     torch.nn.Linear(5*3*270*480, 100),
        #                     torch.nn.ReLU(),
        #                     torch.nn.Linear(100, 3*13*5)
        #                 )

    def InitNet(self):
        # convnet1(with dropout) and loss init
        # an additional ConvNet1
        
        return  torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,      # input height
                        out_channels=8,    # n_filters
                        kernel_size=5,      # filter size
                        stride=2,           # filter movement/step
                        padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                    ),      # output shape (16, 28, 28)
                    nn.Conv2d(8,8,5,1,2),
                    nn.Conv2d(8,1,5,1,2),
                    nn.ReLU(),    # activation
                    nn.MaxPool2d(kernel_size=2),
                    torch.nn.Dropout(0.5),
                )

    def ConvNet2(self):
        # handle the image,multi-layer CNN network
        return  torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,      # input height
                        out_channels=8,    # n_filters
                        kernel_size=5,      # filter size
                        stride=2,           # filter movement/step
                        padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                    ),      # output shape (16, 28, 28)
                    nn.Conv2d(8,8,5,1,2),
                    
                    nn.Conv2d(8,1,5,1,2),      # output shape (16, 28, 28)
                    nn.ReLU(),    # activation
                    nn.MaxPool2d(kernel_size=2),
                )

    def ConvNet3(self):
        # handle lstm output
        return  torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,      # input height
                        out_channels=12,    # n_filters
                        kernel_size=5,      # filter size
                        stride=1,           # filter movement/step
                        padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                    ),      # output shape (16, 28, 28)
                    nn.Conv2d(12,12,5,1,2),
                    nn.Conv2d(12,12,5,1,2),
                    nn.ReLU(),    # activation 134/135 20 12
                    nn.MaxPool2d(kernel_size=2)
                    #67*10 12
                )

                    
    def fuconLoss(self):
        return  torch.nn.Sequential(
                    torch.nn.Linear(67 * 120, 67*120), # 转成图片的大小
                )

    def LSTM(self):
        # LSTM
        return nn.LSTM(input_size=8040, hidden_size=134*20, num_layers=3, batch_first=True)  #  ->   (input_size,hidden_size,num_layers)

    def forward(self, images):  
        # ininet
        #   repeat stages for frame
        out = self.initnet(images[:, 0,:,:,:])
        batchsize, c, h, w = out.shape
        out = out.view(batchsize, c, -1)
        out = self.lossnet(out)
        out = out.view(batchsize, c, h, w)

        gaussianmap = self.genarate_gaussianmap(batchsize).to(self.device)

        hidden = None

        # 每个阶段输入 为 经过图片经过convnet2的输出，central Gaussian map， 上一次loss的输出
        for stage in range(self.stage):
            # print("stage {}:".format(stage))
            input = self.convnet2(images[:,stage,:,:,:])
            # print(out.shape)
            # print(input.shape)
            # print(gaussianmap.shape)
            # print(out.dtype)
            # print(input.dtype)
            # print(gaussianmap.dtype)
            out = torch.cat([out, input, gaussianmap], 1)
            batchsize, _, h, w = out.shape
            out = out.view(batchsize, 3, -1)
            # print(out.shape)

            out, hidden = self.lstm(out, hidden)
            # print(out.shape)

            out = out.view(batchsize, 3, 134, 20)

            out = self.convnet3(out)

            batchsize, c, h, w = out.shape
            out = out.view(batchsize, 1, -1)
            # print(out.shape) # 67*120
            out = self.lossnet(out)
            out = out.view(batchsize, 1, 67, 120)

        out = out.view(batchsize, -1)
        out =self.resultnet(out)
        return out.view(batchsize, 3, 13, self.stage)

    def genarate_gaussianmap(self, batchsize):
        
        seq, c, x, y = np.mgrid[-1.0:1.0:complex(0, batchsize), -1.0:1.0:1j, -1.0:1.0:67j, -1.0:1.0:120j]
        # Need an (N, 2) array of (x, y) pairs.
        xy = np.column_stack([seq.flat, c.flat, x.flat, y.flat])

        mu = np.array([0.0, 0.0, 0.0, 0.0])

        sigma = np.array([.025, .025,.025, .025])
        covariance = np.diag(sigma**2)

        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

        # Reshape back to a (30, 30) grid.
        z = z.reshape(x.shape)

        return torch.FloatTensor(z)


# # print(result.shape) #torch.Size([64, 3, 13, 5])
# # print(label_map.float().shape) #torch.Size([64, 3, 13, 5])
# loss = lossfunc(result, label_map.float())
# # print(loss) # tensor(11093679., device='cuda:0', grad_fn=<MseLossBackward>)
class pmMSELossFunc(nn.Module):
    def __init__(self):
        super(pmMSELossFunc, self).__init__()
        return

    def forward(self, predict, target):
        # batchsize, (x\y\visible), brone, seq
        bs, ddim, actionlen, seqlen = predict.shape

        sum = 0
        MAXLEN = (270 * 270 + 480 * 480) / 4
        for batchsize in range(bs):
            # 每批
            for seq in range(seqlen):
                # 每张图片
                for action in range(actionlen):
                    # 每个动作
                    temp_sum = (target[batchsize][2][action][seq] - predict[batchsize][2][action][seq])**2 *MAXLEN
                    temp_dis = (target[batchsize][1][action][seq] - predict[batchsize][1][action][seq])**2 \
                                    + (target[batchsize][0][action][seq] - predict[batchsize][0][action][seq])**2
                    sum += temp_sum+ temp_dis
        return sum/(bs*seqlen*actionlen)



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