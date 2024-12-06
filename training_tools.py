import sys, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image
from tqdm import trange 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, augmented_transform=None):

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augmented_transform = augmented_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('L')
        label = self.img_labels.iloc[idx, 1]
        sound = self.img_labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        if self.augmented_transform:
            image = self.augmented_transform(image)

        return image, label, sound
        
    
class CustomImageDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):   
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('L')

        if self.transform:
            image = self.transform(image)
            
        return image, img_name
    

class AlexNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(

        nn.Conv2d(num_channels, 48, kernel_size=(11,11), stride=4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

        nn.Conv2d(48, 128, kernel_size=(5,5), stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

        nn.Conv2d(128, 192, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),

        nn.Conv2d(192, 192, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),

        nn.Conv2d(192, 128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        )

        self.classifier = nn.Sequential(

        nn.Flatten(),

        nn.Linear(128*6*6, 512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, num_classes),

        nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        print(x.shape)
        return x
    


def training_loop(epochs, lr, model, data_train, data_val, weight_decay=0):

    model.to(device)

    num_epochs = epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []

    for epoch in trange(num_epochs):
            

        running_loss_train, running_loss_val = 0.0, 0.0
        train_acc = []
        model.train()
        for i, data in enumerate(data_train):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()

            # Calculate accuracy
           ###############################################################
            out = F.softmax(outputs, dim=1)
            cond1 = (out[range(out.size(0)), labels] >= 0.25)
            cond2 = (outputs.argmax(dim=1) == labels)
            acc = torch.logical_or(cond1, cond2).float().mean()
            train_acc.append(acc.item())
           ###############################################################

            
            #preds = outputs.argmax(dim=1)         
            #acc = (preds == labels).float().mean()
            #train_acc.append(acc.item())          
            
            

        loss_train_list.append(running_loss_train / len(data_train))
        acc_train_list.append(np.mean(train_acc))


        val_acc = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_val):    
                inputs, labels, _ = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)       
                running_loss_val += loss.item()

                # Calculate accuracy
               ###############################################################
                out = F.softmax(outputs, dim=1)
                cond1 = (out[range(out.size(0)), labels] >= 0.25)
                cond2 = (outputs.argmax(dim=1) == labels)
                acc = torch.logical_or(cond1, cond2).float().mean()
                val_acc.append(acc.item())
               ###############################################################

               
                #preds = outputs.argmax(dim=1)         
                #acc = (preds == labels).float().mean()
                #val_acc.append(acc.item())            
               
               
        loss_val_list.append(running_loss_val / len(data_val))
        acc_val_list.append(np.mean(val_acc))

        print('Epoch [{}/{}], TRAIN: Loss: {:.4f}, Accuracy: {:.4f} ; TEST: Loss: {:.4f}, Accuracy: {:.4f}'
                    .format(epoch+1, num_epochs, loss_train_list[-1], acc_train_list[-1], loss_val_list[-1], acc_val_list[-1]))

    return loss_train_list, loss_val_list, acc_train_list, acc_val_list



def plot_loss_acc(loss_val_list, loss_train_list, acc_val_list, acc_train_list, figsize=(12, 6)):    
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].plot(loss_val_list, label='val_loss', color="orange")
    ax[0].plot(loss_train_list, label="train_loss", color="royalblue")
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(acc_train_list, label='train_accuracy', color="royalblue")
    ax[1].plot(acc_val_list, label='val_accuracy', color="orange")
    ax[1].set_title('Accuracy')
    ax[1].legend()