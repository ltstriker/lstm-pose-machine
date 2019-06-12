from __future__ import print_function
import argparse
from model.lstmposemachine import *
from data.penn_data import Penn_Data
import torch
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# hyper parameter
temporal = 5
train_data_dir = '/BIGDATA1/nsccgz_yfdu_5/liangt/LSTMPoseMachine/lstm_pm_pytorch/data/Penn_Action/'

# add parameter
parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size for training')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs for training')
parser.add_argument('--models', default='models', type=str, help='directory of checkpoint')
parser.add_argument('--cuda', default=1, type=int, help='use GPU or not')
args = parser.parse_args()

epochs      = args.epochs
batch_size  = args.batch_size
lr          = args.lr

if not os.path.exists(args.models):
    os.mkdir(args.models)

# data
transform = transforms.Compose([transforms.ToTensor()])

# Build dataset
train_data = Penn_Data(data_dir=train_data_dir, transform=transform)
print('Train dataset size:' + str(len(train_data)))
train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

#define model
net = lstmposemachine()

# gpu support
device = torch.device("cpu")
device_ids = [0, 1, 2, 3]
if args.cuda & torch.cuda.is_available():
    # net = net.cuda(device_ids[0])
    device = torch.device("cuda")
    net = nn.DataParallel(net, device_ids=device_ids)

optimizer = optim.Adam(params=net.parameters(), lr=lr)
lossfunc = nn.MSELoss(size_average=True)

def train():
    net.train()
    for epoch in range(epochs):
        print('epoch:' + str(epoch))
        for step, (images, label_map, center_map, imgs) in enumerate(train_dataset):
            images = Variable(images).to(device)
            label_map = Variable(label_map).to(device)
            center_map = Variable(center_map).to(device)
            imgs = Variable(imgs).to(device)

            optimizer.zero_grad()
            result = net(images, center_map)

            total_loss = lossfunc(result, label_map)

            # backward
            total_loss.backward()
            optimizer.step()

            acc = pck_score(result, label_map, 0.2, images.shape)
            

            if step % 10 == 0:
                print('--step .....' + str(step))
                print('--loss ' + str(float(total_loss)))



# do train and test here
if __name__ == '__main__':
    train()