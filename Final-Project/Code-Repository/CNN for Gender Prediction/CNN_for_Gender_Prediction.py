#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import tarfile
import io
import os
# for reading and displaying images
from skimage import io
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
import PIL
import torch
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
import torch.nn.functional as F
from IPython.display import display
#from torchvision.transforms import ToTensor, ToPILImage
# PyTorch libraries and modules
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.optim as optim
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from timeit import default_timer as timer 
import sys
import argparse

os.chdir("/home/jain.he/csye7374-HemantJain/Project")
#get_ipython().run_line_magic('matplotlib', 'inline')

#import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


torch.cuda.device_count()

main_folder = '/home/jain.he/csye7374-HemantJain/Project'
images_folder ='img_align_celeba'
TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
batch_size = 128
epochs = 5

class CelebTrain(Dataset):
    def __init__(self, csv_file,img_dir, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file,nrows = 5120)
        self.transform = transform

    def __len__(self):
        #print(len(self.annotations))
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        #img_name = self.img_path[index]
        img = Image.open(img_path)
        y_label = torch.tensor(int (self.annotations.iloc[index,1]))

        if self.transform:
            img = self.transform(img)

            return (img, y_label)
      

class CelebValid(Dataset):
    def __init__(self, csv_file,img_dir, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file, nrows = 2560)
        self.transform = transform

    def __len__(self):
        #print(len(self.annotations))
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        #img_name = self.img_path[index]
        img = Image.open(img_path)
        y_label = torch.tensor(int (self.annotations.iloc[index,1]))

        if self.transform:
            img = self.transform(img)

            return (img, y_label)
      


# Creating tensors for the images and their labels
train_data = CelebTrain(csv_file = 'train_gender.csv', img_dir = images_folder,
                       transform = transforms.Compose([transforms.Resize(178),transforms.CenterCrop(178)
                                                       ,transforms.ToTensor()]))

valid_data = CelebValid(csv_file = 'valid_gender.csv', img_dir = images_folder,
                       transform = transforms.Compose([transforms.Resize(178),transforms.CenterCrop(178),
                                                        transforms.ToTensor()]))


# loading the data
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle = True)



# Creating the model
class GenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3, stride = 1)
        self.batch1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,5,stride=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,3,stride=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(128, 128, 5, stride=1)
        self.batch4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 8 * 8 ,1024)  
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024,128)
        self.drop2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(128,2)

    def forward(self,x):
        #x1  =   self.conv1(x)
        x   =   self.pool1(F.relu(self.batch1((self.conv1(x)))))
        x   =   self.pool2(F.relu(self.batch2((self.conv2(x)))))
        x   =   self.pool3(F.relu(self.batch3((self.conv3(x)))))
        x   =   self.pool4(F.relu(self.batch4((self.conv4(x)))))
        x   =   x.view(-1, 128 * 8 * 8)
        x   =   self.drop1(x) 
        x   =   F.relu(self.fc1(x))
        x   =   self.drop2(x) 
        x   =   F.relu(self.fc2(x))
        x   =   self.fc3(x)
        return x
print(GenderModel())


model1 = GenderModel()
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model1)

model.to(device)



def train():  # loop over the dataset multiple times
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001, momentum=0.9)   
    criterion = nn.CrossEntropyLoss()
    n_samples = 10240
    total_loss = 0
    total_train = 0
    correct_train = 0.0
    running_loss = 0
    for i,(img, labels) in enumerate(train_loader):
        img = img.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
        #accuracy
        _, predicted = torch.max(output.data, 1)
        predicted = predicted
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train

        
    running_loss += loss.item() * img.size(0) 

    epoch_loss = 100* running_loss/total_train
    losses.append(epoch_loss)
    return (train_accuracy,loss.item())



def valid(epochs):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        with torch.no_grad():
            for i,(img, labels) in enumerate(valid_loader):
                # if i>=n_samp:
                #   break
                img = img.to(device)
                labels = labels.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100*correct/total
    return acc


img_test_dir = 'img_test'

classes = ('Female','Male')
def predict(img_test_dir):
    model.eval()
    #Loading the model
    model1 = torch.load('/model.pt',map_location = 'cpu')
    print(model1)
    #Loadind the test image
    img = Image.open(img_test_dir)
    #print(img)
    with torch.no_grad():
        trans1 = transforms.Compose([transforms.Resize(178),transforms.CenterCrop(178)
                                                       ,transforms.ToTensor()])
        img_tensor = trans1(img) #shape = [3, 32, 32]
        #Image Transformation
        trans = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_tensor = trans(img_tensor)
        print(img_tensor.shape)
        
        single_image_batch = img_tensor.unsqueeze(0) #shape = [1, 3, 32, 32]
        #print(single_image_batch.shape)
        outputs = model1(single_image_batch)
        _, predicted = torch.max(outputs.data, 1)
        class_id = predicted[0].item()
        predicted_class = classes[predicted[0].item()]
        print("Predicted Class : {}".format(predicted_class))
        
            

print("Training for {} epochs".format(epochs))
main_time = timer()
losses = []
for epoch in range(epochs):
    start = timer()
    (train_acc,loss_tr)=train()
    valid_acc = valid(epochs)

    print('Epoch {}, train Loss: {:.3f}'.format(epoch ,loss_tr), 'Training Accuracy {}:' .format(train_acc), 'Valid Accuracy {}'.format(valid_acc))
    print("GPU time: ",timer()-start)
print("Model successfully saved at ./model/")
torch.save(model, 'model.pt')
print("Time taken for",epochs,"epochs is",timer()-main_time,"seconds")

plt.title("Loss plot")
plt.plot(losses)





