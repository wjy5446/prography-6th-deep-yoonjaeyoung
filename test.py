import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from model import *

mnist_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class Model(object):
    def __init__(self):
        self.learning_rate = 0.0001
        self.beta1 = 0.9
        self.beta2 = 0.999
        
        self.batch_size = 4
        self.epochs = 2

    def build_model(self):
        root_dataset = 'MNIST'
        self.train_dataset = MNIST(root_dataset, transform=mnist_transform, train=True, download=True)
        self.test_dataset = MNIST(root_dataset, transform=mnist_transform, train=False, download=True)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        # Model
        self.model = VGG16()

        # Loss
        self.loss = nn.CrossEntropyLoss()

        # Trainer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

    def train(self):
        for epoch in range(self.epochs):
            for idx, (img, label) in enumerate(self.train_loader):
                print(img.size(), label.size())
                # train
                self.optim.zero_grad()
                self.model.train()
                label_pre = self.model(img)
                loss = self.loss(label_pre, label)

                loss.backward()
                self.optim.step()

                print('epoch %5d [%5d/%5d] loss(train): %.8f' % (epoch, idx, len(self.train_loader), loss))
    
    def inference(self, path_img):
        img = Image.open(path_img)
        self.model.eval()
        label = self.model(img)


if __name__ == '__main__':
    m = Model()
    m.build_model()
    m.train()
