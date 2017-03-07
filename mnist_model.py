from __future__ import print_function
import pickle 
import numpy as np
import pandas as pd
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
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import sys

import matplotlib.pyplot as plt
import itertools

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
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


#source of following function: http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def clipped_zoom(img, zoom_factor, **kwargs):
    """
    function to zoom in or out of an image. if we zoom out the image is then padded to the original size
    """

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


#source of following function: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)


def data_augmentation(trainset_labeled):
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


        #apply elastic distortion to image
        #paramter values based off of https://arxiv.org/pdf/1103.4487.pdf
        sigma = np.random.uniform(5,6)
        alpha = np.random.uniform(36,38)
        augmented_dataset.append((torch.from_numpy(np.array([elastic_transform(i[0].numpy()[0],alpha,sigma)])),i[1]))

    return augmented_dataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        #weight initialization
        for m in self.modules():
            #using He et al. initialization method for convolutional layers, based on dimension of inputs
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #using Xavier initialization method for fully connected layers, based on number of input and output units
            elif isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in + n_out)))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2)) #add batch normalization after convolution
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)



def train(model,args,train_loader,unlab_loader,optimizer,epoch, T1, T2, alpha_f):

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


def test(model,args,epoch, valid_loader, test_type):
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
    return 100. * correct / len(valid_loader.dataset)


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


def update_unlabeled(model,args,unlab_loader):
    
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

def save_accuracy_plot(orig_accs,dev_accs,train_accs,epochs):
    #generate accuracy plot
    plt.plot(np.arange(epochs), orig_accs, marker='o', label='Train Accuracy')
    plt.plot(np.arange(epochs), dev_accs, marker='o', label='Validation Accuracy')
    plt.plot(np.arange(epochs), train_accs, marker='o', label='Augmented Accuracy')
    plt.xlabel('Epochs')
    plt.title('MNIST: Train and Validation Accuracies')
    plt.legend(loc='lower right')
    plt.savefig('plots/final_model_accuracies.jpg')



def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


    print('loading data!')
    trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
    validset = pickle.load(open("validation.p", "rb"))
    testset = pickle.load(open("test.p","rb"))
    trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

    augmented_dataset = data_augmentation(trainset_labeled)

    orig_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=64, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    unlab_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=256, shuffle=False)

    #currently the labels for the unlab data are set to None, so it loader wont work
    unlab_loader.dataset.train_labels=torch.LongTensor([0]*len(unlab_loader.dataset)) #initialize these to dummy value


    model = Net()
    if args.cuda:
        model.cuda()

    #using Adam optimizer instead of SGD due to slight increase in performance
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_accs=[]
    dev_accs=[]
    orig_accs=[]
    T1 = 75
    T2 = 125
    alpha_f = 1
    for epoch in range(1, args.epochs + 1):

        '''
        we implement pseudo-labels to update every epoch, starting at epoch = T1 
        '''
        if epoch >= T1: #update pseudolabels for unlabeled data
            update_unlabeled(model, args, unlab_loader)
        
        train(model, args, train_loader, unlab_loader, optimizer, epoch, T1, T2, alpha_f)
        c_train_acc = test(model, args, epoch, train_loader, 'Train')
        c_dev_acc = test(model, args, epoch, valid_loader, 'Dev')
        c_orig_acc = test(model, args, epoch, orig_loader, 'Orig Train')

        dev_accs.append(c_dev_acc) #updates loss for plot 
        train_accs.append(c_train_acc)
        orig_accs.append(c_orig_acc) 

    #save accuracy plots in jpg file
    #save_accuracy_plot(orig_accs,dev_accs,train_accs,args.epochs)

    #save final model
    torch.save(model.state_dict(), "best_model.mdl")


if __name__ == "__main__":
    main()