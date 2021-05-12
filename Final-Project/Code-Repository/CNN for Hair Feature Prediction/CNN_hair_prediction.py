#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
#import tarfile
import io
import os
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# for creating validation set
from sklearn.model_selection import train_test_split
from skimage import io
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
import seaborn as sns

os.chdir("..")
os.chdir("Project/")

# In[ ]:


#import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# In[ ]:


CUDA_LAUNCH_BLOCKING =1


# In[ ]:


main_folder = 'Project/img_align_celeba'
images_folder ='img_align_celeba/img_align_celeba'
TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
batch_size = 128
epochs = 5

train_img = pd.read_csv('df_train_attr.csv')
valid_img = pd.read_csv('df_valid_attr.csv')


# In[ ]:
            

train_img =train_img.dropna()
train_img = train_img.astype({'Black_Hair':'int64','Blond_Hair':'int64','Brown_Hair':'int64'
,'Gray_Hair':'int64','Hair_color':'int64'})


# In[ ]:


valid_img = valid_img.dropna()
valid_img = valid_img.astype({'Black_Hair':'int64','Blond_Hair':'int64','Brown_Hair':'int64'
,'Gray_Hair':'int64','Hair_color':'int64'})


# In[ ]:


class CelebTrain(Dataset):
    def __init__(self, csv_file,img_dir, transform=None):
        self.img_dir = img_dir
        self.annotations = csv_file
        self.transform = transform

    def __len__(self):
        #print(len(self.annotations))
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        #img_name = self.img_path[index]
        img = Image.open(img_path)
        y_label = torch.tensor(int (self.annotations.iloc[index,-1]))

        if self.transform:
            img = self.transform(img)

            return (img, y_label)

class CelebValid(Dataset):
    def __init__(self, csv_file,img_dir, transform=None):
        self.img_dir = img_dir
        self.annotations = csv_file
        self.transform = transform

    def __len__(self):
        #print(len(self.annotations))
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        #img_name = self.img_path[index]
        img = Image.open(img_path)
        y_label = torch.tensor(int (self.annotations.iloc[index,-1]))

        if self.transform:
            img = self.transform(img)

            return (img, y_label)
      


# In[ ]:


# Creating tensors for the images and their labels
train_data = CelebTrain(csv_file = train_img, img_dir = images_folder,
                       transform = transforms.Compose([transforms.Resize(178),transforms.CenterCrop(178)
                                                       ,transforms.ToTensor()]))

valid_data = CelebValid(csv_file = valid_img, img_dir = images_folder,
                       transform = transforms.Compose([transforms.Resize(178),transforms.CenterCrop(178),
                                                        transforms.ToTensor()]))





# In[ ]:


# loading the data
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle = True)


# In[ ]:


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
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024,128)
        self.drop2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(128,4)

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


# In[ ]:


model1 = GenderModel()
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model1)

model.to(device)


# In[ ]:


def train():  # loop over the dataset multiple times
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001, momentum=0.9)   
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_train = 0
    correct_train = 0
    for i,(img, labels) in enumerate(train_loader):
        
        img = img.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        output = model(img)
        # forward + backward + optimize
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

        

    losses.append(loss.item())
    train_acy.append(train_accuracy)
    return (train_accuracy,loss.item())


# In[ ]:


#n_samp = 5120
def valid(epochs):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        with torch.no_grad():
            for i,(img, labels) in enumerate(valid_loader):
                img = img.to(device)
                labels = labels.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100*correct/total
    valid_acy.append(acc)
    return acc


# In[ ]:


classes = ('Blond Hair','Black Hair','Brown Hair','Gray Hair')
def predict(img_test_dir):
    model.eval()
    #Loading the model
    model1 = torch.load('/content/model.pt',map_location = 'cpu')
    #Loading the test image
    img = Image.open(img_test_dir)
    
    with torch.no_grad():
        trans1 = transforms.Compose([transforms.Resize(178),transforms.CenterCrop(178)
                                                       ,transforms.ToTensor()])
        img_tensor = trans1(img) #shape = [3, 32, 32]
        #Image Transformation
        trans = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_tensor = trans(img_tensor)
        print(img_tensor.shape)
        
        single_image_batch = img_tensor.unsqueeze(0) 
        outputs = model1(single_image_batch)
        _, predicted = torch.max(outputs.data, 1)
        class_id = predicted[0].item()
        predicted_class = classes[predicted[0].item()]
        print("Predicted Class : {}".format(predicted_class))
        #x1=x1.squeeze()
        #x1=x1.detach().numpy()
        #fig = plt.figure(figsize=(5, 5))  # width, height in inches
        #display(img)


# In[ ]:


print("Training for {} epochs".format(epochs))
main_time = timer()
losses = []
train_acy = []
valid_acy = []
for epoch in range(epochs):
    start = timer()
    (train_model,loss_tr)=train()
    valid_acc = valid(epochs)

    print('Epoch {}, train Loss: {:.3f}'.format(epoch ,loss_tr), 'Training Accuracy {}:' .format(train_model), 'Valid Accuracy {}'.format(valid_acc))
    print("GPU time: ",timer()-start)
print("Model successfully saved ")
torch.save(model, 'model.pt')
print("Time taken for",epochs,"epochs is",timer()-main_time,"seconds")

epoch = range(1,len(losses)+1)
plt.title("Training Loss plot")
plt.plot(epoch,losses)

plt.figure()
plt.title("Accuracy: train vs valid")
plt.plot(epoch,train_acy,'red',label='Training accuracy')
plt.plot(epoch,valid_acy,'blue',label='Validation accuracy')
plt.legend()
plt.show()

# In[ ]:


predict('img_test/202510.jpg')

