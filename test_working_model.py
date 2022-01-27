import torch
from models.MobileNetV3 import *
from models.ResNet import resnet50
from models.LeNet import *

# LeNet
x = torch.rand(2, 3, 32, 32)
lenet = LeNet(num_classes=9)
out = lenet(x)
print("image shape: ", x[0].shape)
print(out)

print('-'*20)
# MobileNet
x = torch.rand(1, 3, 112, 112)
mobilenet = mobilenet_v3_small(num_classes=2)
out = mobilenet(x)
print("image shape: ", x[0].shape)
print(out)
...
