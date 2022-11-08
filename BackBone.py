import torch
import torch.nn as nn
import torchvision

class BasicBlock(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride,padding,max_pool, down_sample):
        super(BasicBlock,self).__init__()
        self.conv=nn.Conv2d(input_size,output_size,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if max_pool==True:
            self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2, padding=1, dilation=1, ceil_mode=False)
        else:
            self.maxpool=None
        self.relu=nn.LeakyReLU(0.01)
        if down_sample==True:
            self.downsample=nn.Sequential(
                nn.Conv2d(input_size,output_size,kernel_size=(1,1),stride=2),
                nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        else:
            self.down_sample=None

    def forward(self,x):
        identity=x
        x=self.conv(x)
        x=self.bn(x)

        if self.maxpool is not None:
            x=self.maxpool(x)

        if self.downsample is not None:
            identity=self.downsample(identity)
        x+=identity

        x=self.relu(x)

        return x


class ShallowResNet(nn.Module):
    def __init__(self):
        super(ShallowResNet,self).__init__()
        self.conv1=nn.Conv2d(1,4,(5,5),2,1)
        self.bn1=nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu=nn.LeakyReLU(0.01)

        self.layer2=BasicBlock(input_size=4,output_size=4,kernel_size=(3,3),stride=1,padding=1,max_pool=True,down_sample=True)
        self.layer3=BasicBlock(input_size=4,output_size=8,kernel_size=(3,3),stride=2,padding=1,max_pool=False,down_sample=True)
        self.layer4=BasicBlock(input_size=8,output_size=8,kernel_size=(3,3),stride=1,padding=1,max_pool=True,down_sample=True)
        self.layer5=BasicBlock(input_size=8,output_size=16,kernel_size=(3,3),stride=2,padding=1,max_pool=False,down_sample=True)
        self.layer6=BasicBlock(input_size=16,output_size=16,kernel_size=(3,3),stride=1,padding=1,max_pool=True,down_sample=True)
        self.layer7=BasicBlock(input_size=16,output_size=16,kernel_size=(3,3),stride=2,padding=1,max_pool=False,down_sample=True)

        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(5,5))

        self.classifier=nn.Sequential(
            nn.Linear(5*5*16,5*5*2),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.5,inplace=False),

            #nn.Linear(5*5*8,5*5*4),
            #nn.LeakyReLU(0.01),
            #nn.Dropout(p=0.5,inplace=False),

            nn.Linear(5*5*2,2)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.maxpool(x)
        x-self.relu(x)

        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x=self.layer7(x)
        
        x=self.avgpool(x)

        x=x.view(-1,self.num_flat_features(x))
        x=self.classifier(x)

        return x
    
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s

        return num_features