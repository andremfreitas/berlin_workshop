# your imports

import torch
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as T
from torchvision.transforms.v2 import CenterCrop
from typing import Optional, Callable
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#from tqdm import tqdm
import h5py
from PIL import Image
import pandas as pd
from torchvision.transforms import v2

# your datasets

# make sure data is moved to the GPU
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

# mean and std for computation. Obtained previously in ex1.
mean = [0.6371044628053755, 0.6373997169466824, 0.6370696801635732]
std = [0.25507978734128617, 0.2546555985632842, 0.25461099284149014]

# transform = v2.Compose([v2.ToDtype(torch.float32, scale=True),
#     v2.ToTensor(),  # Convert to PyTorch tensor
#     v2.Normalize(mean = mean, std = std),  # Replace with actual mean and std
# ])

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),  # Convert to PyTorch tensor
    v2.Normalize(mean = mean, std = std)  # Replace with actual mean and std
])

class H5Dataset(Dataset):
    def __init__(self, file_path_d,file_path_t, transform):
        self.file_d = h5py.File(file_path_d, 'r')
        self.data = self.file_d['x']
        self.file_t = h5py.File(file_path_t, 'r')
        self.target = self.file_t['y']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (torch.from_numpy(self.data[idx]).permute((2,0,1)),torch.from_numpy(self.target[idx].squeeze()))

        if self.transform:
            sample = self.transform(sample)
        return sample


train_dataset = H5Dataset("/tmp/pcam/camelyonpatch_level_2_split_train_x.h5","/tmp/pcam/camelyonpatch_level_2_split_train_y.h5", transform = transform)
val_dataset = H5Dataset("/tmp/pcam/camelyonpatch_level_2_split_valid_x.h5","/tmp/pcam/camelyonpatch_level_2_split_valid_y.h5", transform = transform)

# create DataLoaders
batch_size = 256

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# your solution
from torch import nn
import torch.nn.functional as F

class CNN_LR(torch.nn.Module):
    def __init__(self):
        super(CNN_LR, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 12) # (in_channels, out_channels, kernel size, stride, padding, ...) stride, default = 1, padding default is 0
        # flatten the output of the conv layer
        self.fc = torch.nn.Linear(1*85*85, 1) # size of linear layer = image width - kernel size + padding -- I think this formula is wrong.
        # the formula above I think should be = /(image width - kernel size + 2*padding) / stride) + 1

    def forward(self, x):
        x = self.conv(x)
        # flatten the output of the conv layer
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3) 
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,  1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = x.view(-1, self.num_flat_features(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(64, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return torch.sigmoid(out)

    # def num_flat_features(self, x):
    #     size = x.size()[1:]
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

#model = CNN_LR()

class ConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts, k, s, p):
        super(ConvBlock, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.convolution(input_img)

        return x


class ReduceConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts_1, out_fts_2, k, p):
        super(ReduceConvBlock, self).__init__()
        self.redConv = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts_1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_fts_1, out_channels=out_fts_2, kernel_size=(k, k), stride=(1, 1), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.redConv(input_img)

        return x


class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 128, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.avgpool(input_img)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class InceptionModule(nn.Module):
    def __init__(self, curr_in_fts, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvBlock(curr_in_fts, f_1x1, 1, 1, 0)
        self.conv2 = ReduceConvBlock(curr_in_fts, f_3x3_r, f_3x3, 3, 1)
        self.conv3 = ReduceConvBlock(curr_in_fts, f_5x5_r, f_5x5, 5, 2)

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=curr_in_fts, out_channels=f_pool_proj, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def forward(self, input_img):
        out1 = self.conv1(input_img)
        out2 = self.conv2(input_img)
        out3 = self.conv3(input_img)
        out4 = self.pool_proj(input_img)

        x = torch.cat([out1, out2, out3, out4], dim=1)

        return x

class MyGoogleNet(nn.Module):
    def __init__(self, in_fts=3, num_class=1):
        super(MyGoogleNet, self).__init__()
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 192, 3, 1, 1)
        )

        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.aux_classifier1 = AuxClassifier(512, num_class)
        self.aux_classifier2 = AuxClassifier(528, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024 * 7 * 7, num_class)
        )

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out2 = self.aux_classifier2(x)
        x = self.inception_4e(x)
        x = self.maxpool1(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        if self.training == True:
            return [x, out1, out2]
        else:
            return x

model = MyGoogleNet()
model = model.to('cuda')



def calc_accuracy(model, data_loader):
    # evaluate val loss
    acc = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.unsqueeze(1).float().to(device)
        y_pred = model(x)
        acc += ((y_pred > 0.5) == y).sum().item()
    acc /= (len(data_loader)*data_loader.batch_size)
    return acc

def check_cuda(model):
    if next(model.parameters()).is_cuda:
        print("Model is on CUDA device.")
    else:
        print("Error: Model is not on CUDA device. Please move the model to CUDA.")
        # You can also move the model to CUDA using the following line:
        # model.to('cuda')

check_cuda(model)
val_acc_old = calc_accuracy(model, val_loader)
print("The validation accuracy of the untrained model is: {}".format(val_acc_old))

# initialise the model, define the optimizer and loss

learning_rate = 1e-3
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# create an LR scheduler, dividing the learning rate by 10 every epoch
# lr_lambda = lambda epoch: 0.1
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
print("Start training...")

epochs = 1

for epoch in range(epochs):
    print("Epoch", epoch+1)
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.unsqueeze(1).float().to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {(batch_idx+1)}: Train Loss {loss.item()}")
    print()
    # Update the learning rate
    # scheduler.step()

    #train_acc = calc_accuracy(model, train_loader)
    val_acc = calc_accuracy(model, val_loader)
    #print("Train accuracy:", train_acc)
    print("Validation accuracy:", val_acc)

    # if val_acc > val_acc_old:
    #     print("Validation accuracy improved from {} to {}".format(val_acc_old, val_acc))
    #     print("Continue training")
    #     val_acc_old = val_acc
    # else:
    #     print("Validation accuracy did not improve")
    #     print("Stop training")
    #     break
    print()
"""
class HDF5Dataset(Dataset):
    def __init__(self, input_file_path, transforms=None):
        self.input_file_path = input_file_path
        self.input_file = h5py.File(self.input_file_path, 'r')
        self.input = self.input_file['x']
        self.transform = transform


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x = self.input[idx]

        # Convert the numpy array to a PIL image
        x_pil = Image.fromarray(x)

        # Apply the specified transform if provided
        if self.transform:
            x_pil = self.transform(x_pil)
        return x_pil
 # Number of CPU workers for data loading
# Usage

test_dataset = HDF5Dataset(
    input_file_path='/tmp/pcam/camelyonpatch_level_2_split_test_x.h5', transforms=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def save_test_predictions(model, test_loader):
    test_preds = []
    for x in test_loader:
        x = x.to("cuda")
        y_pred = model(x).round().to(torch.int).cpu()
        test_preds.append(y_pred)
    result_df = pd.DataFrame(torch.stack(test_preds).reshape(-1).numpy(), columns=["predicted label"])
    result_df.to_csv("predicted_labels_test_alexnet.csv", index=False)
save_test_predictions(model, test_loader)
"""