import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from model import *

from PIL import Image

mnist_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class Model(object):
    def __init__(self):
        self.learning_rate = 0.0001
        self.beta1 = 0.9
        self.beta2 = 0.999
        
        self.batch_size = 16
        self.epochs = 2

    def build_model(self):
        root_dataset = 'MNIST'
        self.train_dataset = MNIST(root_dataset, transform=mnist_transform, train=True, download=True)
        self.test_dataset = MNIST(root_dataset, transform=mnist_transform, train=False, download=True)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = VGG16().to(self.device)
        

        # Loss
        self.loss = nn.CrossEntropyLoss().to(self.device)

        # Trainer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

    def train(self):
        for epoch in range(self.epochs):
            for idx, (img, label) in enumerate(self.train_loader):
                img, label = img.to(self.device), label.to(self.device)
                # train
                self.optim.zero_grad()
                self.model.train()
                label_pre = self.model(img)
                loss = self.loss(label_pre, label)

                loss.backward()
                self.optim.step()

                if idx % 100 == 0:
                    self.save('model', idx)
                print('epoch %5d [%5d/%5d] loss(train): %.8f' % (epoch, idx, len(self.train_loader), loss))

    def test(self, idx_load_param):

        self.load('model', idx_load_param)

        self.model.eval()

        num_correct = 0
        for idx, (img, label) in enumerate(self.test_loader):
            img, label = img.to(self.device), label.to(self.device)
            label_pre = self.model(img)
            digit_pre = torch.argmax(label_pre, dim=-1).long()

            num_correct += torch.sum(digit_pre == label).float()
            print('precision: {}'.format(num_correct / float((idx + 1) * self.batch_size)))

        print('final precision: {}'.format(num_correct / len(self.test_dataset)))

    def inference(self, path_img, idx_load_param):
        img = Image.open(path_img)
        img = mnist_transform(img).unsqueeze(0)
        img = img.to(self.device)

        self.load('model', idx_load_param)
        self.model.eval()
        label = self.model(img)
        digit = torch.argmax(label)
        print(digit)

    def save(self, dir, step):
        param = self.model.state_dict()
        torch.save(param, os.path.join(dir, 'param_%07d.pt') % step)

    def load(self, dir, step):
        param = torch.load(os.path.join(dir, 'param_%07d.pt') % step)
        self.model.load_state_dict(param)

import argparse
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='inference')
    parser.add_argument('--idx_load_param', type=int, default=2000)
    parser.add_argument('--path_img', type=str, default='data/img5_gray.jpg')
    args = parser.parse_args()

    m = Model()
    m.build_model()   

    mode = args.phase 

    if mode == 'train':
        m.train()
    if mode == 'test':
        m.test(args.idx_load_param)
    if mode == 'inference':
        m.inference(args.path_img, args.idx_load_param)
