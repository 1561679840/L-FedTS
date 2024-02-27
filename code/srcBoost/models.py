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
        self.loss_fn = nn.NLLLoss()   #SVM loss

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

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

class Fmnist_BNNet(nn.Module):

    def __init__(self):
        super(Fmnist_BNNet,self).__init__()
        self.conv1 = nn.Conv2d(1,64,1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        self.fc6 = nn.Linear(512,10)
        self.loss_fn = nn.NLLLoss()

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        #print(" x shape ",x.size())
        x = x.view(-1,128*8*8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return F.log_softmax(x, dim=1)






