from torch import nn
import torch
import torch.nn.functional as F

def hinge_loss(outputs, labels):
    """
    SVM折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    return loss


class Mnist_Model(nn.Module):
    #CNN
    def __init__(self):
        super(Mnist_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Boston_Model(nn.Module):
    #MLP
    def __init__(self):
        super(Boston_Model, self).__init__()
        self.fc1 = nn.Linear(13, 14)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(14, 5)
        self.loss_fn = nn.NLLLoss()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class KDD_Model(nn.Module):
    #SVM
    def __init__(self):
        super(KDD_Model, self).__init__()
        self.fc = nn.Linear(41, 23)
        self.loss_fn = hinge_loss   #SVM loss

    def forward(self, x):
        x = self.fc(x)
        return x

class CIFAR_Model_ZFNet(nn.Module):
    def __init__(self):
        super(CIFAR_Model_ZFNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,7,2,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(96,128,5,2,0),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.classifier=nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,10),
        )
        self.loss_fn = nn.NLLLoss()
    def forward(self,x):
        x=self.features(x)
        x=x.view(-1,256*6*6)
        x=self.classifier(x)
        return F.log_softmax(x, dim=1)

class CIFAR_Model_LeNet(nn.Module):
    def __init__(self):
        super(CIFAR_Model_LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.relu=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.maxpool2=nn.MaxPool2d(2,2)

        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        self.loss_fn = nn.NLLLoss()

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.maxpool2(x)
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return F.log_softmax(x, dim=1)


