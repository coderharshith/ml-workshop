import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5),(0.5))])

#load the dataset
train_dataset = MNIST(root='./data',train=True,download=True,transform=trans)
test_dataset = MNIST(root='./data',train=False,download=True,transform=trans)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)


class simplenet(nn.Module):
    def __init__(self):
        super(simplenet,self).__init__()
        self.fc1=nn.Linear(28*28,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,128)
        self.fc4=nn.Linear(128,10)
    
    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x) 
        x=F.relu(x)
        x=self.fc3(x) 
        x=F.relu(x)
        x=self.fc4(x) 
        return x

model = simplenet()

criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

num_ephoch = 10

for epoch in range(num_ephoch):
    runing_loss = 0.0
    for image,lable in train_loader:
        output =model(image)
        loss = criterian(output,lable)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        runing_loss+=loss.item()
    print(f'Epoch:- {epoch+1} loss:{runing_loss/len(train_loader)}')


model.eval()
with torch.no_grad():
    correct = 0.0
    total = 0.0
    for image,label in test_loader:
        output =  model(image)
        _,predicted =  torch.max(output, 1)
        correct += (predicted==label).sum().item()
        total += label.size(0)
    print(f'accuracy:{(correct/total)*100}')