from __future__ import print_function
import pickle 
import numpy as np
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.legacy.optim as legacy_optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import scipy.misc
import scipy.ndimage
import sys

import matplotlib.pyplot as plt
import itertools

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)



print('loading data!')
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
validset = pickle.load(open("validation.p", "rb"))
testset = pickle.load(open("test.p","rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

#source of following function: http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = scipy.ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


#Data augmentation
augmented_dataset = []
for i in trainset_labeled:
    #add original image
    augmented_dataset.append(i)

    #rotate image
    rotation_degree = np.random.uniform(-45,45)
    augmented_dataset.append((torch.from_numpy(np.array([scipy.ndimage.rotate(i[0].numpy()[0],rotation_degree,reshape=False)])),i[1]))
    
    #shift image (translation in both x and y axes)
    shift_amount = np.random.uniform(-2,2)
    augmented_dataset.append((torch.from_numpy(np.array([scipy.ndimage.shift(i[0].numpy()[0],shift_amount)])),i[1]))

    #zoom in or out of image

    #choose magnitude of amount to either zoom in or zoom out
    zoom_amount = np.random.uniform(1,2)
    
    #pick either 0 or 1, like a binary coin flip
    random_draw = np.random.randint(0,2)

    if random_draw==0: #zoom out
        zoom_amount = 1/zoom_amount
    else: #zoom in 
        zoom_amount = 1 * zoom_amount

    augmented_dataset.append((torch.from_numpy(np.array([clipped_zoom(i[0].numpy()[0],zoom_amount)])),i[1]))


train_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=64, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
unlab_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=256, shuffle=True)

#currently the labels for the unlab data are set to None, so it loader wont work
unlab_loader.dataset.train_labels=torch.LongTensor([-1]*len(unlab_loader.dataset)) #initialize these to dummy value


# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        #Xavier weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in + n_out)))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(epoch, T1, T2, alpha_f):

    model.train()
    #izip stops when the shorter one (train_loader) is exhausted. can tweak batch sizes (line 138) to include all data
    for (batch_idx, (data, target)), (unlab_data, unlab_target) in itertools.izip(enumerate(train_loader),unlab_loader):
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            unlab_data, unlab_target = unlab_data.cuda(), unlab_target.cuda()

        data, target = Variable(data), Variable(target)
        unlab_data, unlab_target = Variable(unlab_data), Variable(unlab_target)

        optimizer.zero_grad()
        
        output = model(data)
        output_unlab = model(unlab_data)

        loss = (F.nll_loss(output, target) + pseudo_weight(epoch, T1=T1, T2=T2, alpha_f=alpha_f)*F.nll_loss(output_unlab,unlab_target)) #apply weighted pseudo
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch, valid_loader, test_type):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += (F.nll_loss(output, target).data[0]) #don't apply pseudo here
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\n' + test_type + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return test_loss


def pseudo_weight(t,T1=100,T2=600,alpha_f=3):

    '''
    EQN 16 from: http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf
    
    Default values for T1, T2 and alpha_f are from paper
    '''

    if t < T1:
        return 0
    elif t >= T2:
        return alpha_f 
    else: #T1 <= t < T2
        return alpha_f*(t-T1)/float(T2-T1) 


def update_unlabeled():
    
    model.eval() #set model in eval mode  

    for idx, (data, target) in enumerate(unlab_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        pred = output.data.max(1)[1]

        if len(pred)==unlab_loader.batch_size:
            unlab_loader.dataset.train_labels[idx*unlab_loader.batch_size:(idx+1)*unlab_loader.batch_size] = pred
        else:
            unlab_loader.dataset.train_labels[idx*unlab_loader.batch_size:] = pred
    
    #print (unlab_loader.dataset.train_labels)
    #print (sum([i==-1 for i in unlab_loader.dataset.train_labels]))


train_accs=[]
dev_accs=[]
T1 = 1
T2 = 6
alpha_f = 3
for epoch in range(1, args.epochs + 1):

    '''
    we implement pseudo-labels to update every epoch, starting at epoch = T1 
    '''
    if epoch >= T1: #update pseudolabels for unlabeled data
        update_unlabeled()
    
    train(epoch, T1, T2, alpha_f)
    c_train_acc = test(epoch, train_loader, 'Train')
    c_dev_acc = test(epoch, valid_loader, 'Dev')

    dev_accs.append(c_dev_acc) #updates loss for plot 
    train_accs.append(c_train_acc) 


plt.plot(np.arange(args.epochs), dev_accs, marker='o', label='Validation Accuracy')
plt.plot(np.arange(args.epochs), train_accs, marker='o', label='Train Accuracy')
plt.title('MNIST: Train and Validation Losses')
plt.legend(loc='upper left')
plt.savefig('losses.jpg')
