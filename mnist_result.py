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
from mnist_model import Net

#load model class and the saved model parameters
model = Net()
model.load_state_dict(torch.load("best_model.mdl"))

#load test data
testset = pickle.load(open("test.p","rb"))
test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

#generate predictions
label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))

#restructure predictions into index - ID dataframe
predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)

#save predictions
predict_label.to_csv('final_submission.csv', index=False)
