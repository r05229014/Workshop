from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torchsummary import summary
from tqdm import tqdm
# Setting hyperparameters

BATCH_SIZE = 64 
IMAGE_SIZE = 64

# Creating the transformations

transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
    ])

class ManhattenDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform



# Loading the dataset 
dataset = dset.DatasetFolder(root = '../Manhattan/', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(dataloader):
        print(batch_x.numpy().shape, batch_y.numpy())
