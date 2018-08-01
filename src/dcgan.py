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
import numpy as np
# Setting hyperparameters

BATCH_SIZE = 64 
IMAGE_SIZE = 64

# Creating the transformations

transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.Grayscale(),
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,)),
    ])

# Loading the dataset 
dataset = dset.ImageFolder(root = '../Manhattan/', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
                    nn.ConvTranspose2d(100,512,4,1,0, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(512,256,4,2,1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256,128,4,2,1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128,64,4,2,1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64,1,4,2,1, bias =False),
                     nn.Tanh()
                )  
    def forward(self, input):
        output = self.main(input)
        return output


class D(nn.Module):

    def __init__(self):
        super(D,self).__init__()
        self.main = nn.Sequential(
                    nn.Conv2d(1,64,4,2,1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64,128,4,2,1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128,256,4,2,1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256,512,4,2,1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(512,1,4,1,0, bias=False),
                      nn.Sigmoid()
                )  

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)



netG = G()
netG.cuda()
netG.apply(weights_init)
summary(netG,input_size=(100,1,1))


netD = D()
netD.cuda()
netD.apply(weights_init)
summary(netD,input_size=(1,64,64,))


criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

loss_D = []
loss_G = []
for epoch in range(1000):
    
    for i, data in tqdm(enumerate(dataloader, 0)):
        # Train Discriminator
        #print(data.shape)
        netD.zero_grad()

        real, _ = data # real data
        #print(real.shape)
        input = Variable(real).cuda()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(input)
        errD_real = criterion(output, target)

        noise = Variable(torch.randn(input.size()[0], 100,1,1)).cuda() # fake data
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0])).cuda()
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        loss_D.append(errD.data[0])
        loss_G.append(errG.data[0])
        #print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 300, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 ==0:
            #print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 300, i, len(dataloader), errD.data[0], errG.data[0]))
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
loss_D = np.array(loss_D)
loss_G = np.array(loss_G)


