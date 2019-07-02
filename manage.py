from __future__ import print_function
import argparse
from model.lstmposemachine import *
from data.penn_data import Penn_Data

import os
import torch
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# hyper parameter
temporal = 5
train_data_dir = '/BIGDATA1/nsccgz_yfdu_5/liangt/LSTMPoseMachine/Penn_Action/'

# add parameter
parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size for training')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs for training')
parser.add_argument('--cuda', default=1, type=int, help='use GPU or not')
args = parser.parse_args()

epochs      = args.epochs
batch_size  = args.batch_size
lr          = args.lr

# data
transform = transforms.Compose([transforms.Resize([ 270, 480]),transforms.ToTensor()])

# Build dataset
train_data = Penn_Data(data_dir=train_data_dir, transform=transform)
print('Train dataset size:' + str(len(train_data)))
train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


# gpu support
device = torch.device("cpu")
device_ids = [0,1,2,3]

if args.cuda & torch.cuda.is_available():
    device = torch.device("cuda")
    #define model
    net = lstmposemachine(out=7, stage=3, device=device)
    net = net.cuda(device_ids[0])
    net = nn.DataParallel(net, device_ids=device_ids)
else:
    #define model
    net = lstmposemachine(out=7, stage=3, device=device)

print(device)

optimizer = optim.Adam(params=net.parameters(), lr=lr)
# lossfunc = nn.MSELoss(reduction='mean')
lossfunc = pmMSELossFunc()

def train():
    net.train()
    for epoch in range(epochs):
        print('\n\nepoch:{}'.format(epoch))
        tot_loss = []
        tot_acc = []
        for step, (images, label_map) in enumerate(train_dataset):
            images = Variable(images).to(device)
            label_map = Variable(label_map).to(device)
            # print(images.shape)
            # print(label_map.shape)

            optimizer.zero_grad()
            result = net(images)

            # print(result.shape) #torch.Size([64, 3, 13, 5])
            # print(label_map.float().shape) #torch.Size([64, 3, 13, 5])
            loss = lossfunc(result, label_map.float())
            # print(loss) # tensor(11093679., device='cuda:0', grad_fn=<MseLossBackward>)

            tot_loss.append(loss)

            # backward
            loss.backward()
            optimizer.step()

            _,_,_,h,w = images.shape
            acc = pck_score(result, label_map, 0.2, (h,w))
            
            tot_acc.append(acc[-1])
            # if step == 0:
            print('--step:{} loss{:.2f} acc{:.2f}'.format(step, loss, acc[-1] ))
        torch.save(net.state_dict(), './params/densnet_params_{}.pt'.format(epoch))    
        print('\n--epoch:{} loss{:.2f} acc{:.2f}'.format(epoch, sum(tot_loss)/len(tot_loss), sum(tot_acc)/len(tot_acc)))

# do train and test here
if __name__ == '__main__':
    train()